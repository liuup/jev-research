"""Generation-wise online RLCD training on the 24 game.

One generation is: freeze the current policy, roll out the training puzzles under it,
keep one observed outcome per ``(state, action)``, train the head on those rows, then
evaluate on the dev split. The learner never sees a label that its own ongoing updates
produced, because collection finishes before the first optimizer step of a generation.
"""

import argparse
import json
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from .config import load_config
from .evaluation import (
    observed_metrics,
    oracle_separation,
    plot_reliability,
    predict_rows,
    risk_coverage,
)
from .model import JevModel, load_tokenizer
from .optimization import optimization_step
from .puzzles import load_oracle, load_puzzles, verify_manifest
from .rollout import collect, record_json, rollout_batch, score_candidates
from .runtime import atomic_checkpoint, load_checkpoint, require_slurm
from .serialization import question_row
from .utils import rate, save_json, seed_all, stream_seed, write_jsonl

INTEGER_KEYS = (
    "generations",
    "puzzles_per_generation",
    "rollouts_per_puzzle",
    "steps_per_generation",
    "effective_batch",
    "microbatch",
    "warmup_steps",
    "validation_puzzles",
    "test_puzzles",
    "inference_questions",
    "reward_samples",
    "max_length",
)


def validate_config(c):
    for key in INTEGER_KEYS:
        if not isinstance(c[key], int) or c[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if c["microbatch"] > c["effective_batch"]:
        raise ValueError("microbatch must not exceed effective_batch")
    if c["objective"] not in ("paired_pg", "ce", "brier"):
        raise ValueError(f"Unknown objective: {c['objective']}")
    if c["baseline"] not in ("conditional", "zero"):
        raise ValueError(f"Unknown baseline: {c['baseline']}")
    if c["inference_precision"] not in ("fp32", "bf16"):
        raise ValueError("inference_precision must be fp32 or bf16")
    if c["rollout_mode"] not in ("sample", "greedy"):
        raise ValueError("rollout_mode must be sample or greedy")
    if c["init_policy"] not in ("model", "solver_mix"):
        raise ValueError("init_policy must be model or solver_mix")
    if not 0.0 <= c["init_solver_mix"] <= 1.0:
        raise ValueError("init_solver_mix must lie in [0, 1]")
    if c["temperature"] <= 0:
        raise ValueError("temperature must be positive")


def sample_puzzles(puzzles, count, rng):
    if count > len(puzzles):
        return rng.choices(puzzles, k=count)
    return rng.sample(puzzles, count)


def generation_policy_id(config, generation):
    """Prompt label for the policy that continues an episode in this generation."""
    base = f"24game:seed{config['seed']}"
    if generation == 0 and config["init_policy"] == "solver_mix" and config["init_solver_mix"] > 0:
        return f"{base}:solver{config['init_solver_mix']:g}:g0"
    return f"{base}:g{generation}"


def generation_config(config, generation):
    """The frozen policy of one generation: its prompt label and its controller mix."""
    mix = config["init_solver_mix"] if (
        generation == 0 and config["init_policy"] == "solver_mix"
    ) else 0.0
    return dict(config, policy_id=generation_policy_id(config, generation), solver_mix=mix)


def evaluate_policy(model, tokenizer, puzzles, config, mode="greedy", oracle=None):
    """Roll out puzzles under the current policy and score the visited rows."""
    ordered, seeds = [], []
    for index, puzzle in enumerate(puzzles):
        ordered.append(puzzle)
        seeds.append(stream_seed(config["seed"], "eval", config["policy_id"], puzzle["puzzle_id"], index))
    records = rollout_batch(model, tokenizer, ordered, config, seeds, mode=mode)
    rows = []
    for record in records:
        outcome = "yes" if record["solved"] else "no"
        for step in record["steps"]:
            rows.append(
                question_row(step["state"], step["action"], config["policy_id"])
                | dict(observed_outcome=outcome, controller=step["controller"])
            )
    metrics = dict(
        puzzles=len(puzzles),
        mode=mode,
        solved=sum(record["solved"] for record in records),
        solved_rate=rate(sum(record["solved"] for record in records), len(records)),
        mean_steps=float(np.mean([len(record["steps"]) for record in records])),
        mean_action_entropy=float(
            np.mean([step["entropy"] for record in records for step in record["steps"]])
        ),
        mean_p_yes_chosen=float(
            np.mean([step["p_yes"] for record in records for step in record["steps"]])
        ),
    )
    if oracle is not None:
        first = [record["steps"][0] for record in records if record["steps"]]
        metrics["first_action_solvable_rate"] = rate(
            sum(
                oracle.get((record["puzzle"]["puzzle_id"], 0), {}).get(step["chosen"], False)
                for record, step in zip(
                    [record for record in records if record["steps"]], first
                )
            ),
            len(first),
        )
        metrics["oracle_states"] = len(first)
    if rows:
        predictions = predict_rows(
            model,
            tokenizer,
            rows,
            microbatch=config["inference_questions"],
            max_length=config["max_length"],
            precision=config["inference_precision"],
        )
        metrics |= observed_metrics(predictions, rows)
        metrics |= risk_coverage(predictions, rows)
        if oracle is not None:
            metrics["oracle_separation"] = oracle_separation(predictions, rows, oracle)
    return records, rows, metrics


def run(config, output, model=None, tokenizer=None):
    validate_config(config)
    if model is None:
        require_slurm()
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Use a fresh run directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    seed_all(config["seed"])
    data_root = Path(config["data_dir"])
    manifest = verify_manifest(data_root)
    train_puzzles = load_puzzles(data_root / "train.jsonl")
    dev_puzzles = load_puzzles(data_root / "dev.jsonl")[: config["validation_puzzles"]]
    oracle = load_oracle(data_root)
    if not train_puzzles or not dev_puzzles:
        raise ValueError("The generated dataset is empty")

    if model is None:
        model = JevModel.load_base(config).float().cuda()
    if tokenizer is None:
        tokenizer = load_tokenizer(config["base_model"])
    device = next(model.parameters()).device
    if device.type == "cuda":
        require_slurm()
    config = dict(config)
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
    save_json(
        output / "resolved_config.json",
        dict(
            config=config,
            manifest=manifest,
            train_puzzles=len(train_puzzles),
            dev_puzzles=len(dev_puzzles),
            git_commit=subprocess.run(
                ["git", "rev-parse", "HEAD"], text=True, capture_output=True
            ).stdout.strip(),
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
            generation_started = time.monotonic()
            frozen = generation_config(config, generation)
            rng = random.Random(stream_seed(config["seed"], "puzzles", generation))
            puzzles = sample_puzzles(train_puzzles, config["puzzles_per_generation"], rng)
            records, rows, rollout_metrics = collect(
                model,
                tokenizer,
                puzzles,
                frozen,
                generation,
                mode=config["rollout_mode"],
            )
            write_jsonl(folder / "trajectories.jsonl", [record_json(r) for r in records])
            write_jsonl(folder / "rows.jsonl", rows)
            save_json(folder / "rollout_metrics.json", rollout_metrics)
            if not rows:
                raise RuntimeError("Generation produced no training rows")
            batch_rng = random.Random(stream_seed(config["seed"], "batch", generation))
            step_records = []
            for step in range(config["steps_per_generation"]):
                batch_rows = [
                    rows[index]
                    for index in batch_rng.choices(
                        range(len(rows)), k=config["effective_batch"]
                    )
                ]
                record, micro = optimization_step(
                    model, tokenizer, batch_rows, optimizer, scheduler, config, micro
                )
                record["step"] = generation * config["steps_per_generation"] + step
                record["generation"] = generation
                step_records.append(record)
                log.write(json.dumps(record) + "\n")
                log.flush()
                print(json.dumps(record), flush=True)
            dev_records, dev_rows, dev_metrics = evaluate_policy(
                model, tokenizer, dev_puzzles, frozen, mode="greedy", oracle=oracle
            )
            write_jsonl(folder / "dev_rows.jsonl", dev_rows)
            save_json(folder / "dev_metrics.json", dev_metrics)
            plot_reliability(dev_metrics, f"{output.name}_gen{generation:03d}")
            summary = dict(
                generation=generation,
                policy_id=frozen["policy_id"],
                solver_mix=frozen["solver_mix"],
                rollout=rollout_metrics,
                dev={key: value for key, value in dev_metrics.items() if key != "events"},
                mean_loss=float(np.mean([record["loss"] for record in step_records])),
                mean_gradient_norm=float(
                    np.mean([record["gradient_norm"] for record in step_records])
                ),
                generation_seconds=time.monotonic() - generation_started,
                total_seconds=time.monotonic() - started,
            )
            summaries.append(summary)
            save_json(output / "generations.json", summaries)
            atomic_checkpoint(
                output / "checkpoint.pt",
                model,
                dict(config, policy_id=frozen["policy_id"]),
                optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(),
                step=(generation + 1) * config["steps_per_generation"],
                generation=generation,
                metadata=dict(device=device.type, dev=summary["dev"]),
            )
            print(json.dumps(dict(generation_complete=summary)), flush=True)
    finally:
        log.close()
    save_json(
        output / "summary.json",
        dict(
            generations=len(summaries),
            train_puzzles=len(train_puzzles),
            dev_puzzles=len(dev_puzzles),
            final_dev=summaries[-1]["dev"] if summaries else None,
            best_dev_solved_rate=max(
                (summary["dev"]["solved_rate"] for summary in summaries), default=None
            ),
            seconds=time.monotonic() - started,
        ),
    )
    return summaries


def learning_rate_scale(step, total_steps, warmup_steps):
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    remaining = max(0, total_steps - step)
    return remaining / max(1, total_steps - warmup_steps)


def evaluate_test(run, model=None, tokenizer=None):
    """Final evaluation on the test split and the unsolvable robustness set."""
    require_slurm()
    output = Path(run)
    model, saved = load_checkpoint(output / "checkpoint.pt")
    config = dict(saved["config"])
    tokenizer = tokenizer if tokenizer is not None else load_tokenizer(config["base_model"])
    data_root = Path(config["data_dir"])
    oracle = load_oracle(data_root)
    report = {}
    for name, filename, limit in (
        ("test", "test.jsonl", config["test_puzzles"]),
        ("unsolvable", "unsolvable.jsonl", None),
    ):
        puzzles = load_puzzles(data_root / filename)
        if limit is not None:
            puzzles = puzzles[:limit]
        records, rows, metrics = evaluate_policy(
            model, tokenizer, puzzles, config, mode="greedy", oracle=oracle
        )
        metrics["reached_target_rate"] = rate(
            sum(record["solved"] for record in records), len(records)
        )
        metrics["solver_step_share"] = rate(
            sum(step["controller"] == "solver" for record in records for step in record["steps"]),
            sum(len(record["steps"]) for record in records),
        )
        write_jsonl(output / f"{name}_trajectories.jsonl", [record_json(r) for r in records])
        write_jsonl(output / f"{name}_rows.jsonl", rows)
        save_json(output / f"{name}_metrics.json", metrics)
        report[name] = {key: value for key, value in metrics.items() if key != "events"}
    save_json(output / "test_report.json", report)
    return report


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/smoke.yaml")
    parser.add_argument("--model-config", default="configs/model.yaml")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=YAML_VALUE")
    parser.add_argument("--output", required=True)
    parser.add_argument("--test", action="store_true", help="evaluate the test splits")
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
