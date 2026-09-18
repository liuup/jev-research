"""Online generalized policy iteration with versioned terminal feedback."""

import copy
import json
import os
import random
import subprocess
import time
from pathlib import Path

import torch

from .env import Game
from .evaluation import mc_metrics, observed_metrics, plot_reliability, predict
from .model import JevModel, load_tokenizer
from .online import (
    AsyncOnlineCollector,
    OnlineCollector,
    mc_reference,
    retarget_questions,
    stream_seed,
    terminal_feedback,
    validate_events,
)
from .online_evaluation import action_ranking_metrics, play_seeded, promotion_decision
from .optimization import optimization_step
from .policies import FrozenHeuristic, restore_policy, snapshot_candidate
from .runtime import checkpoint, load_checkpoint, require_slurm
from .utils import digest, save_json


def validate_online_config(c):
    for key in (
        "max_policy_generations",
        "min_optimizer_steps_per_policy",
        "max_optimizer_steps_per_policy",
        "promotion_interval_steps",
        "max_total_optimizer_steps",
        "effective_batch",
        "microbatch",
        "source_envs",
        "select_every",
        "inference_questions",
        "validation_questions",
        "eval_every",
        "mc_states",
        "mc_rollouts",
        "promotion_games",
        "test_games",
        "game_batch",
    ):
        if not isinstance(c[key], int) or c[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if not isinstance(c["prefetch_batches"], int) or c["prefetch_batches"] < 0:
        raise ValueError("prefetch_batches must be a nonnegative integer")
    if c["microbatch"] > c["effective_batch"] or c["effective_batch"] % c["microbatch"]:
        raise ValueError("Microbatch must divide effective_batch")
    if c["mc_rollouts"] < 2 or c["reward_samples"] < 2:
        raise ValueError("MC and paired PG require at least two samples")
    if not 0 <= c["behavior_epsilon"] <= 1:
        raise ValueError("Invalid behavior epsilon")
    if (
        c["min_optimizer_steps_per_policy"]
        > c["max_optimizer_steps_per_policy"]
        or c["max_total_optimizer_steps"] < c["min_optimizer_steps_per_policy"]
    ):
        raise ValueError("Invalid per-policy optimizer-step bounds")
    if c["promotion_min_log_tile_gain"] < 0 or c["warmup_steps"] < 0:
        raise ValueError("Negative margin or warmup")
    if c["objective"] not in ("ce", "brier", "paired_pg"):
        raise ValueError("Unknown objective")
    if c["utility"] != "log_tile" or c["threshold"] not in (
        256,
        512,
        1024,
        2048,
        4096,
        8192,
    ):
        raise ValueError("Online policy improvement requires expected log-tile utility")
    if "data_dir" in c:
        raise ValueError("Online mode must not depend on an offline data directory")


def write_jsonl(path, rows):
    with Path(path).open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def probability_evaluation(model, tokenizer, c, questions, policy, generation, output):
    validation = terminal_feedback(
        retarget_questions(
            questions, policy.policy_id, generation, c["seed"], "validation"
        ),
        policy,
    )
    write_jsonl(output / "validation.jsonl", validation)
    p = predict(model, tokenizer, validation, c["inference_questions"], c["max_length"])
    metrics = observed_metrics(p, validation)
    save_json(
        output / "initial_validation_predictions.json",
        [
            dict(
                state_id=r["state_id"],
                action=r["action"],
                observed_outcome=r["observed_outcome"],
                p=v.tolist(),
            )
            for r, v in zip(validation, p)
        ],
    )
    return validation, metrics


def run_online(
    c, output, model=None, tokenizer=None, incumbent=None, game_factory=Game
):
    validate_online_config(c)
    if model is None:
        require_slurm()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(f"Use a fresh online run directory: {output}")
    if model is None:
        if c.get("initial_checkpoint"):
            model, saved = load_checkpoint(c["initial_checkpoint"])
            for key in ("base_model", "head_width", "attention_heads"):
                if saved["config"][key] != c[key]:
                    raise ValueError(f"Initial checkpoint mismatch: {key}")
            del saved
        else:
            model = JevModel.load_base(c).float().cuda()
    tokenizer = tokenizer if tokenizer is not None else load_tokenizer(c["base_model"])
    incumbent = (
        incumbent if incumbent is not None else FrozenHeuristic(c["heuristic_config"])
    )
    device = next(model.parameters()).device
    if device.type == "cuda":
        require_slurm()
    checkpoint(output / "initial.pt", model, c, kind="online_initialization")
    metadata = dict(
        config=c,
        mode="online",
        device_type=device.type,
        model_class=type(model).__name__,
        slurm_job_id=os.environ.get("SLURM_JOB_ID"),
        torch_version=torch.__version__,
        initial_sha256=digest(output / "initial.pt"),
        initial_checkpoint=c.get("initial_checkpoint") or None,
        initial_policy=incumbent.spec,
        git_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        git_dirty=bool(
            subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        ),
        versioning="one frozen continuation policy per generation; no cross-generation replay",
        simulator_sha256=digest(Path(__file__).with_name("env.py")),
        policy_source_sha256=digest(Path(__file__).with_name("policies.py")),
    )
    save_json(output / "resolved_config.json", metadata)
    reference_collector = OnlineCollector(incumbent, c, -1, "holdout", game_factory)
    questions = reference_collector.collect_questions(c["validation_questions"])
    write_jsonl(output / "reference_questions.jsonl", questions)
    # Holdout boards/actions stay fixed; labels must be regenerated for each target policy.
    state_ids = list(dict.fromkeys(row["state_id"] for row in questions))
    selected = set(
        random.Random(c["seed"]).sample(state_ids, min(c["mc_states"], len(state_ids)))
    )
    mc_questions = [row for row in questions if row["state_id"] in selected]
    global_step = 0
    micro = c["microbatch"]
    summaries = []
    cumulative = dict(behavior_steps=0, rollout_steps=0, terminal_rollouts=0)
    with (output / "training.jsonl").open("w") as log:
        generation = 0
        while (
            generation < c["max_policy_generations"]
            and global_step + c["min_optimizer_steps_per_policy"]
            <= c["max_total_optimizer_steps"]
        ):
            target_policy = incumbent
            print(
                json.dumps(
                    dict(
                        event="generation_start",
                        generation=generation,
                        target_policy_id=target_policy.policy_id,
                    )
                ),
                flush=True,
            )
            folder = output / f"generation_{generation:03d}"
            folder.mkdir()
            target_spec = copy.deepcopy(target_policy.spec)
            save_json(folder / "target_policy.json", target_spec)
            async_collection = (
                c["prefetch_batches"] > 0
                and target_policy.spec.get("kind") == "heuristic"
                and game_factory is Game
                and device.type == "cuda"
            )
            collector = (
                AsyncOnlineCollector(
                    target_policy,
                    c,
                    generation,
                    c["effective_batch"],
                    c["prefetch_batches"],
                )
                if async_collection
                else OnlineCollector(target_policy, c, generation, "train", game_factory)
            )
            print(
                json.dumps(
                    dict(
                        event="collector_start",
                        generation=generation,
                        mode="async_process" if async_collection else "synchronous",
                        prefetch_batches=c["prefetch_batches"] if async_collection else 0,
                    )
                ),
                flush=True,
            )
            validation, initial_metrics = probability_evaluation(
                model, tokenizer, c, questions, target_policy, generation, folder
            )
            save_json(folder / "initial_validation_metrics.json", initial_metrics)
            # A policy generation owns one optimizer. Rejected candidates continue with
            # the same moments and schedule; a promoted target starts a fresh optimizer.
            optimizer = torch.optim.AdamW(
                [
                    dict(params=model.backbone.parameters(), lr=c["backbone_lr"]),
                    dict(params=model.head.parameters(), lr=c["head_lr"]),
                ],
                weight_decay=c["weight_decay"],
                foreach=False,
            )

            def schedule(step):
                if step < c["warmup_steps"]:
                    return (step + 1) / max(1, c["warmup_steps"])
                return max(
                    0.0,
                    (c["max_optimizer_steps_per_policy"] - step)
                    / max(
                        1,
                        c["max_optimizer_steps_per_policy"] - c["warmup_steps"],
                    ),
                )

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            train_start = time.monotonic()
            events_file = None
            candidate = None
            decision = None
            old_games = None
            new_games = None
            policy_step = 0
            try:
                events_file = (folder / "events.jsonl").open("w")
                while (
                    policy_step < c["max_optimizer_steps_per_policy"]
                    and global_step < c["max_total_optimizer_steps"]
                ):
                    collect_start = time.monotonic()
                    rows = collector.collect(c["effective_batch"])
                    collect_seconds = time.monotonic() - collect_start
                    validate_events(rows, target_policy.policy_id, generation)
                    for row in rows:
                        events_file.write(json.dumps(row) + "\n")
                    events_file.flush()
                    record, micro = optimization_step(
                        model, tokenizer, rows, optimizer, scheduler, c, micro
                    )
                    global_step += 1
                    policy_step += 1
                    record.update(
                        step=global_step,
                        generation=generation,
                        policy_step=policy_step,
                        collection_seconds=collect_seconds,
                        producer_collection_seconds=getattr(
                            collector, "producer_seconds", collect_seconds
                        ),
                        online_questions_per_second=len(rows)
                        / (collect_seconds + record["step_seconds"]),
                        environment=dict(collector.counters),
                        cumulative_environment={
                            k: cumulative[k] + collector.counters[k] for k in cumulative
                        },
                    )
                    record.pop("step_seconds", None)
                    record.pop("questions_per_second", None)
                    at_policy_limit = (
                        policy_step == c["max_optimizer_steps_per_policy"]
                        or global_step == c["max_total_optimizer_steps"]
                    )
                    if policy_step % c["eval_every"] == 0 or at_policy_limit:
                        metrics = observed_metrics(
                            predict(
                                model,
                                tokenizer,
                                validation,
                                c["inference_questions"],
                                c["max_length"],
                            ),
                            validation,
                        )
                        save_json(
                            folder / "dev" / f"step_{policy_step:06d}.json", metrics
                        )
                        record["dev"] = {
                            k: v for k, v in metrics.items() if k != "events"
                        }
                    log.write(json.dumps(record) + "\n")
                    log.flush()
                    print(json.dumps(record), flush=True)

                    eligible = policy_step >= c["min_optimizer_steps_per_policy"]
                    scheduled = (
                        policy_step == c["min_optimizer_steps_per_policy"]
                        or (
                            policy_step > c["min_optimizer_steps_per_policy"]
                            and (
                                policy_step - c["min_optimizer_steps_per_policy"]
                            )
                            % c["promotion_interval_steps"]
                            == 0
                        )
                    )
                    if not eligible or not (scheduled or at_policy_limit):
                        continue
                    saved_config = dict(c, microbatch=micro)
                    checkpoint(
                        folder / "checkpoint.pt",
                        model,
                        saved_config,
                        kind="online_generation",
                        generation=generation,
                        policy_step=policy_step,
                        step=global_step,
                        target_policy=target_spec,
                        optimizer=optimizer.state_dict() if c["save_optimizer"] else None,
                        scheduler=scheduler.state_dict(),
                        torch_rng=torch.get_rng_state(),
                        cuda_rng=torch.cuda.get_rng_state()
                        if device.type == "cuda"
                        else None,
                        metadata=metadata,
                    )
                    candidate = snapshot_candidate(
                        model,
                        tokenizer,
                        folder / "checkpoint.pt",
                        c,
                        target_policy.policy_id,
                    )
                    promotion_seeds = [
                        stream_seed(c["seed"], generation, policy_step, "promotion", i)
                        for i in range(c["promotion_games"])
                    ]
                    old_games = play_seeded(
                        target_policy,
                        promotion_seeds,
                        c["game_batch"],
                        game_factory,
                        progress=dict(
                            generation=generation,
                            policy_step=policy_step,
                            phase="promotion",
                            role="incumbent",
                        ),
                    )
                    new_games = play_seeded(
                        candidate,
                        promotion_seeds,
                        c["game_batch"],
                        game_factory,
                        progress=dict(
                            generation=generation,
                            policy_step=policy_step,
                            phase="promotion",
                            role="candidate",
                        ),
                    )
                    decision = promotion_decision(
                        old_games,
                        new_games,
                        c["promotion_min_log_tile_gain"],
                    )
                    promotion_report = dict(
                        policy_step=policy_step,
                        decision=decision,
                        incumbent=old_games,
                        candidate=new_games,
                    )
                    save_json(
                        folder / f"promotion_step_{policy_step:06d}.json",
                        promotion_report,
                    )
                    save_json(folder / "promotion.json", promotion_report)
                    if decision["accepted"] or at_policy_limit:
                        break
                    del candidate
                    candidate = None
            finally:
                collector.close()
                if events_file is not None:
                    events_file.close()
            if candidate is None or decision is None:
                raise RuntimeError("Policy phase ended without an eligible promotion check")
            training_stats = dict(
                training_seconds=time.monotonic() - train_start,
                actual_microbatch=micro,
                peak_gpu_gib=torch.cuda.max_memory_allocated() / 2**30
                if device.type == "cuda"
                else 0,
                peak_gpu_reserved_gib=torch.cuda.max_memory_reserved() / 2**30
                if device.type == "cuda"
                else 0,
            )
            for key in cumulative:
                cumulative[key] += collector.counters[key]

            final_p = predict(
                model, tokenizer, validation, c["inference_questions"], c["max_length"]
            )
            metrics = observed_metrics(final_p, validation)
            save_json(
                folder / "final_validation_predictions.json",
                [
                    dict(
                        state_id=row["state_id"],
                        action=row["action"],
                        observed_outcome=row["observed_outcome"],
                        p=probability.tolist(),
                    )
                    for row, probability in zip(validation, final_p)
                ],
            )
            mc_rows, mc_steps = mc_reference(
                retarget_questions(
                    mc_questions,
                    target_policy.policy_id,
                    generation,
                    c["seed"],
                    "mc",
                ),
                target_policy,
                c["mc_rollouts"],
                stream_seed(c["seed"], generation, "mc_repeats"),
                c["inference_questions"],
            )
            write_jsonl(folder / "mc_reference.jsonl", mc_rows)
            mc_p = predict(
                model, tokenizer, mc_rows, c["inference_questions"], c["max_length"]
            )
            save_json(folder / "mc_predictions.json", mc_p.tolist())
            metrics.update(mc_metrics(mc_p, mc_rows))
            metrics["action_ranking"] = action_ranking_metrics(
                mc_p, mc_rows, c["utility"], c["threshold"]
            )
            metrics.update(
                target_policy_id=target_policy.policy_id,
                generation=generation,
                policy_step=policy_step,
                mc_environment_steps=mc_steps,
            )
            save_json(folder / "metrics.json", metrics)
            plot_reliability(metrics, f"{output.name}_g{generation:03d}")
            save_json(folder / "candidate_policy.json", candidate.spec)
            # Reporting games are never used by the acceptance rule.
            test_seeds = [
                stream_seed(c["seed"], "test_games", i) for i in range(c["test_games"])
            ]
            test_old = play_seeded(
                target_policy, test_seeds, c["game_batch"], game_factory,
                progress=dict(generation=generation, phase="test", role="incumbent"),
            )
            test_new = play_seeded(
                candidate, test_seeds, c["game_batch"], game_factory,
                progress=dict(generation=generation, phase="test", role="candidate"),
            )
            save_json(
                folder / "controller_test.json",
                dict(
                    incumbent=test_old,
                    candidate=test_new,
                    interpretation="Closed-loop policy performance, separate from frozen-policy probability metrics",
                ),
            )
            if decision["accepted"]:
                incumbent = candidate
            save_json(folder / "active_policy.json", incumbent.spec)
            save_json(output / "active_policy.json", incumbent.spec)
            summary = dict(
                generation=generation,
                objective=c["objective"],
                seed=c["seed"],
                target_policy_id=target_spec["policy_id"],
                active_policy_id=incumbent.policy_id,
                accepted=decision["accepted"],
                status="promoted" if decision["accepted"] else "policy_stalled",
                optimizer_steps=policy_step,
                probability={k: v for k, v in metrics.items() if k != "events"},
                candidate_test={k: v for k, v in test_new.items() if k != "records"},
                incumbent_test={k: v for k, v in test_old.items() if k != "records"},
                environment=cumulative.copy(),
                training_stats=training_stats,
            )
            summaries.append(summary)
            save_json(output / "generations.json", summaries)
            print(json.dumps(dict(generation_complete=summary)), flush=True)
            del optimizer, scheduler, collector
            if not decision["accepted"]:
                break
            generation += 1
    return summaries


def reevaluate_online(run):
    root = Path(run)
    model, saved = load_checkpoint(root / "checkpoint.pt")
    if saved.get("kind") != "online_generation":
        raise ValueError("Expected an online generation checkpoint")
    c = saved["config"]
    target_spec = saved["target_policy"]
    tokenizer = load_tokenizer(c["base_model"])
    policy = restore_policy(target_spec, tokenizer, c)
    rows = [
        json.loads(line)
        for line in (root / "validation.jsonl").read_text().splitlines()
    ]
    if any(r["policy_id"] != policy.policy_id for r in rows):
        raise ValueError("Validation target mismatch")
    metrics = observed_metrics(
        predict(model, tokenizer, rows, c["inference_questions"], c["max_length"]), rows
    )
    mc = [
        json.loads(line)
        for line in (root / "mc_reference.jsonl").read_text().splitlines()
    ]
    if any(r["policy_id"] != policy.policy_id for r in mc):
        raise ValueError("MC target mismatch")
    p = predict(model, tokenizer, mc, c["inference_questions"], c["max_length"])
    metrics.update(mc_metrics(p, mc))
    metrics["action_ranking"] = action_ranking_metrics(
        p, mc, c["utility"], c["threshold"]
    )
    save_json(root / "reevaluated_metrics.json", metrics)
    return metrics
