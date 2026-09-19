"""Generation-wise online training: freeze a policy, collect rollouts, train, repeat."""

import argparse
import hashlib
import json
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from .config import load_config
from .env import State
from .evaluation import (
    action_ranking,
    observed_metrics,
    plot_reliability,
    predict_rows,
    risk_coverage,
)
from .gsm8k import load_rows, split_rows
from .model import JevModel, load_tokenizer
from .optimization import optimization_step
from .rollout import collect, record_json, rollout_batch, score_candidates
from .runtime import atomic_checkpoint, load_checkpoint, require_slurm
from .serialization import question_row
from .utils import rate, save_json, seed_all, write_jsonl

INTEGER_KEYS = (
    "generations",
    "problems_per_generation",
    "rollouts_per_problem",
    "max_steps",
    "steps_per_generation",
    "effective_batch",
    "microbatch",
    "eval_every",
    "checkpoint_every",
    "validation_problems",
    "inference_questions",
    "reward_samples",
    "max_length",
    "max_quantities",
)


def stream_seed(seed, *parts):
    message = ":".join(map(str, (seed, *parts)))
    return int.from_bytes(hashlib.sha256(message.encode()).digest()[:8], "big") % (2**63 - 1)


def validate_config(c):
    for key in INTEGER_KEYS:
        if not isinstance(c[key], int) or c[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if c["microbatch"] > c["effective_batch"]:
        raise ValueError("microbatch must not exceed effective_batch")
    if c["objective"] not in ("paired_pg", "ce", "brier"):
        raise ValueError("Unknown objective")
    if c["baseline"] not in ("conditional", "zero"):
        raise ValueError("Unknown paired-PG baseline")
    if c["rollout_mode"] not in ("sample", "greedy"):
        raise ValueError("Unknown rollout mode")
    if c["inference_precision"] not in ("fp32", "bf16"):
        raise ValueError("Unknown inference precision")
    if c["temperature"] <= 0:
        raise ValueError("temperature must be positive")
    if not 0 < c["validation_fraction"] < 1:
        raise ValueError("validation_fraction must lie in (0, 1)")
    if c["reward_samples"] < 2:
        raise ValueError("paired PG needs at least two samples")


def learning_rate_scale(step, steps, warmup_steps):
    if step < warmup_steps:
        return (step + 1) / max(1, warmup_steps)
    return max(0.0, (steps - step) / max(1, steps - warmup_steps))


def sample_problems(rows, count, rng):
    if count > len(rows):
        raise ValueError("Not enough problems for the requested generation size")
    return rng.sample(rows, count)


def rollout_seeds(problems, config, generation, R):
    return {
        (problem["id"], repeat): stream_seed(
            config["seed"], generation, problem["id"], repeat
        )
        for problem in problems
        for repeat in range(R)
    }


def evaluate_policy(model, tokenizer, config, problems, generation, mode="greedy"):
    seeds = [
        stream_seed(config["seed"], "validation", generation, problem["id"])
        for problem in problems
    ]
    records = rollout_batch(model, tokenizer, problems, config, seeds, mode)
    rows = []
    for record in records:
        problem = next(item for item in problems if item["id"] == record["problem_id"])
        for step in record["steps"]:
            candidate = next(
                item for item in step["state"].candidates() if item["key"] == step["chosen"]
            )
            rows.append(
                question_row(step["state"], candidate, config["policy_id"])
                | dict(
                    observed_outcome="yes" if record["correct"] else "no",
                    problem_id=record["problem_id"],
                )
            )
    metrics = dict(
        problems=len(problems),
        final_answer_accuracy=rate(sum(r["correct"] for r in records), len(records)),
        all_actions_correct_rate=rate(
            sum(all(step["kind"] == "STOP" for step in r["steps"]) for r in records),
            len(records),
        ),
        forced_stop_rate=rate(sum(r["forced"] for r in records), len(records)),
        mean_steps=float(np.mean([len(r["steps"]) for r in records])),
        mode=mode,
    )
    if rows:
        predictions = predict_rows(
            model,
            tokenizer,
            rows,
            config["inference_questions"],
            config["max_length"],
            config["inference_precision"],
        )
        metrics.update(observed_metrics(predictions, rows))
        metrics.update(risk_coverage(predictions, rows))
    return records, rows, metrics


def step_action_reference(model, tokenizer, config, problem, repeats, generation):
    """Empirical success rate of every first-step action, by repeated frozen rollouts."""
    state = State.initial(problem)
    first = state.candidates()
    reference, model_probabilities = {}, {}
    probabilities = score_candidates(model, tokenizer, [(state, first)], config)[0]
    seeds, problems = [], []
    for index, candidate in enumerate(first):
        for repeat in range(repeats):
            seeds.append(
                stream_seed(config["seed"], "action-ref", generation, problem["id"], candidate["key"], repeat)
            )
            problems.append(problem)
    records = rollout_batch(model, tokenizer, problems, config, seeds)
    for index, candidate in enumerate(first):
        chunk = records[index * repeats : (index + 1) * repeats]
        reference[candidate["key"]] = rate(sum(r["correct"] for r in chunk), len(chunk))
        model_probabilities[candidate["key"]] = probabilities[candidate["key"]]
    return model_probabilities, reference, records


def run(config, output, model=None, tokenizer=None, rows=None):
    validate_config(config)
    if model is None:
        require_slurm()
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Use a fresh run directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    seed_all(config["seed"])
    data_root = Path(config["data_dir"])
    if rows is None:
        rows = load_rows(data_root / "train.jsonl")
    rows = [row for row in rows if len(row["numbers"]) <= config["max_quantities"]]
    if not rows:
        raise ValueError("No problem survives the quantity cap")
    rl_rows, validation_rows = split_rows(
        rows, config["validation_fraction"], config["split_seed"]
    )
    if model is None:
        model = JevModel.load_base(config).float().cuda()
    if tokenizer is None:
        tokenizer = load_tokenizer(config["base_model"])
    device = next(model.parameters()).device
    if device.type == "cuda":
        require_slurm()
    config = dict(config)
    config["policy_id"] = f"jevmath:seed{config['seed']}"
    optimizer = torch.optim.AdamW(
        [
            dict(params=model.backbone.parameters(), lr=config["backbone_lr"]),
            dict(params=model.head.parameters(), lr=config["head_lr"]),
        ],
        weight_decay=config["weight_decay"],
        foreach=False,
    )
    total_steps = config["generations"] * config["steps_per_generation"]
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: learning_rate_scale(step, total_steps, config["warmup_steps"])
    )
    validation_problems = validation_rows[: config["validation_problems"]]
    save_json(
        output / "resolved_config.json",
        dict(
            config=config,
            data_manifest=(
                json.loads((data_root / "manifest.json").read_text())
                if (data_root / "manifest.json").exists()
                else None
            ),
            rl_train_problems=len(rl_rows),
            validation_problems=len(validation_rows),
            git_commit=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
        ),
    )
    (output / "resolved_config.yaml").write_text(yaml.safe_dump(config))
    micro = config["microbatch"]
    summaries = []
    started = time.monotonic()
    log = (output / "training.jsonl").open("w")
    try:
        for generation in range(config["generations"]):
            folder = output / f"generation_{generation:03d}"
            folder.mkdir()
            generation_config = dict(
                config, policy_id=f"{config['policy_id']}:g{generation}"
            )
            rng = random.Random(stream_seed(config["seed"], "sample", generation))
            problems = sample_problems(
                rl_rows, config["problems_per_generation"], rng
            )
            seeds = rollout_seeds(
                problems, generation_config, generation, config["rollouts_per_problem"]
            )
            collect_started = time.monotonic()
            records, training_rows = collect(
                model,
                tokenizer,
                problems,
                generation_config,
                seeds,
                mode=config["rollout_mode"],
            )
            collect_seconds = time.monotonic() - collect_started
            if not training_rows:
                raise RuntimeError("Generation produced no training rows")
            write_jsonl(folder / "trajectories.jsonl", [record_json(r) for r in records])
            write_jsonl(folder / "rows.jsonl", training_rows)
            rollout_metrics = dict(
                generation=generation,
                problems=len(problems),
                rollouts=len(records),
                training_rows=len(training_rows),
                final_answer_accuracy=rate(
                    sum(r["correct"] for r in records), len(records)
                ),
                forced_stop_rate=rate(sum(r["forced"] for r in records), len(records)),
                mean_steps=float(np.mean([len(r["steps"]) for r in records])),
                mean_action_entropy=float(
                    np.mean([step["entropy"] for r in records for step in r["steps"]])
                ),
                collect_seconds=collect_seconds,
            )
            save_json(folder / "rollout_metrics.json", rollout_metrics)
            batch_rng = random.Random(stream_seed(config["seed"], "batch", generation))
            step_records = []
            for step in range(config["steps_per_generation"]):
                batch_rows = [
                    training_rows[index]
                    for index in batch_rng.choices(
                        range(len(training_rows)), k=config["effective_batch"]
                    )
                ]
                record, micro = optimization_step(
                    model,
                    tokenizer,
                    batch_rows,
                    optimizer,
                    scheduler,
                    config,
                    micro,
                )
                record["step"] = generation * config["steps_per_generation"] + step
                record["generation"] = generation
                step_records.append(record)
                log.write(json.dumps(record) + "\n")
                log.flush()
                print(json.dumps(record), flush=True)
            validation_records, validation_rows_used, validation_metrics = evaluate_policy(
                model,
                tokenizer,
                generation_config,
                validation_problems,
                generation,
                mode="greedy",
            )
            action_preview = None
            if config["action_reference_problems"] and validation_problems:
                model_probabilities, reference, _ = step_action_reference(
                    model,
                    tokenizer,
                    generation_config,
                    validation_problems[0],
                    config["action_reference_repeats"],
                    generation,
                )
                action_preview = action_ranking([model_probabilities], [reference])
            summary = dict(
                generation=generation,
                policy_id=generation_config["policy_id"],
                rollout=rollout_metrics,
                validation={k: v for k, v in validation_metrics.items() if k != "events"},
                action_ranking=action_preview,
                mean_loss=float(np.mean([r["loss"] for r in step_records])),
                mean_gradient_norm=float(
                    np.mean([r["gradient_norm"] for r in step_records])
                ),
                seconds=time.monotonic() - started,
            )
            write_jsonl(folder / "validation_rows.jsonl", validation_rows_used)
            save_json(folder / "validation_metrics.json", summary["validation"])
            plot_reliability(
                validation_metrics, f"{output.name}_gen{generation:03d}"
            )
            summaries.append(summary)
            save_json(output / "generations.json", summaries)
            atomic_checkpoint(
                output / "checkpoint.pt",
                model,
                config,
                optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(),
                step=(generation + 1) * config["steps_per_generation"],
                generation=generation,
                metadata=dict(device=device.type),
            )
            print(json.dumps(dict(generation_complete=summary)), flush=True)
    finally:
        log.close()
    save_json(
        output / "summary.json",
        dict(
            generations=summaries,
            training_seconds=time.monotonic() - started,
            peak_gpu_gib=torch.cuda.max_memory_allocated() / 2**30
            if device.type == "cuda"
            else 0.0,
        ),
    )
    return summaries


def evaluate_test(run, model=None, tokenizer=None):
    """Final evaluation of a trained checkpoint on the official GSM8K test split."""
    require_slurm()
    output = Path(run)
    model, saved = load_checkpoint(output / "checkpoint.pt")
    config = dict(saved["config"])
    tokenizer = tokenizer if tokenizer is not None else load_tokenizer(config["base_model"])
    problems = load_rows(Path(config["data_dir"]) / "test.jsonl")
    picks = random.Random(config["seed"]).sample(
        problems, min(config["test_problems"], len(problems))
    )
    records, rows, metrics = evaluate_policy(
        model, tokenizer, config, picks, generation=999, mode="greedy"
    )
    write_jsonl(output / "test_trajectories.jsonl", [record_json(r) for r in records])
    write_jsonl(output / "test_rows.jsonl", rows)
    save_json(output / "test_metrics.json", metrics)
    return metrics


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/smoke.yaml")
    parser.add_argument("--model-config", default="configs/model.yaml")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=YAML_VALUE")
    parser.add_argument("--output", required=True)
    parser.add_argument("--test", action="store_true", help="evaluate the test split")
    args = parser.parse_args()
    config = load_config(args.model_config) | load_config(args.config)
    for item in args.set:
        key, value = item.split("=", 1)
        if key not in config:
            raise ValueError(f"Unknown config key: {key}")
        config[key] = yaml.safe_load(value)
    return args, config


def main():
    args, config = arguments()
    if args.test:
        print(json.dumps(evaluate_test(args.output), indent=2, default=str))
        return
    summaries = run(config, args.output)
    print(json.dumps(dict(generations=len(summaries)), indent=2))
