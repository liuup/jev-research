"""Pure-online policy iteration for short-horizon Snake event distributions."""

import json
import os
import time
from pathlib import Path

import torch

from .collector import OnlineCollector
from .env import SnakeGame
from .evaluation import (
    action_ranking_metrics,
    mc_metrics,
    mc_reference,
    play_seeded,
    promotion_decision,
)
from .events import OUTCOMES, validate_event_config
from .inference import observed_metrics, predict
from .model import JevSnakeModel, load_tokenizer
from .optimization import optimization_step
from .policies import UniformPolicy, snapshot_candidate
from .runtime import checkpoint, load_checkpoint, require_slurm
from .utils import digest, save_json, stream_seed


def validate_online_config(config):
    validate_event_config(config)
    integer_keys = (
        "seed",
        "board_size",
        "initial_length",
        "episode_horizon",
        "max_policy_generations",
        "min_optimizer_steps_per_policy",
        "max_optimizer_steps_per_policy",
        "promotion_interval_steps",
        "max_total_optimizer_steps",
        "source_envs",
        "select_every",
        "effective_batch",
        "microbatch",
        "warmup_steps",
        "eval_every",
        "reward_samples",
        "inference_questions",
        "validation_events",
        "mc_states",
        "mc_rollouts",
        "promotion_games",
        "test_games",
        "game_batch",
    )
    for key in integer_keys:
        if not isinstance(config[key], int) or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if config["seed"] != 25:
        raise ValueError("Project experiments require seed 25")
    if not 0 <= config["behavior_epsilon"] <= 1:
        raise ValueError("behavior_epsilon must be in [0,1]")
    if config["objective"] not in ("ce", "brier", "paired_pg"):
        raise ValueError("objective must be ce, brier, or paired_pg")
    if config["baseline"] not in ("conditional", "zero"):
        raise ValueError("baseline must be conditional or zero")
    if config["microbatch"] > config["effective_batch"]:
        raise ValueError("microbatch cannot exceed effective_batch")
    if config["effective_batch"] % config["microbatch"]:
        raise ValueError("microbatch must divide effective_batch")
    if config["min_optimizer_steps_per_policy"] > config[
        "max_optimizer_steps_per_policy"
    ]:
        raise ValueError("Invalid per-policy optimizer bounds")
    if config["max_total_optimizer_steps"] < config[
        "min_optimizer_steps_per_policy"
    ]:
        raise ValueError("Total optimizer budget is too small")
    if config["initial_length"] > config["board_size"]:
        raise ValueError("initial_length cannot exceed board_size")


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, allow_nan=False) + "\n")


def action_questions(validation, config):
    states, seen = [], set()
    for row in validation:
        key = row["state_id"]
        if key in seen:
            continue
        seen.add(key)
        states.append(row)
        if len(states) >= config["mc_states"]:
            break
    questions = []
    for row in states:
        game = SnakeGame.from_state(
            {
                "seed": 0,
                "size": row["size"],
                "rng_state": 0,
                "body": row["body"],
                "direction": row["direction"],
                "food": row["food"],
                "score": row["score"],
                "steps": row["steps"],
                "terminal": False,
                "outcome": None,
            }
        )
        for action in game.legal_actions:
            questions.append(
                {
                    key: value
                    for key, value in row.items()
                    if key not in ("action", "observed_outcome", "event_steps")
                }
                | {
                    "state_id": row["state_id"],
                    "action": action,
                    "candidate_ids": list(OUTCOMES),
                }
            )
    return questions


def _model(config):
    if config.get("initial_checkpoint"):
        model, saved = load_checkpoint(config["initial_checkpoint"])
        for key in ("base_model", "head_width", "attention_heads"):
            if saved["config"][key] != config[key]:
                raise ValueError(f"Initial checkpoint mismatch: {key}")
        if tuple(saved["config"]["outcomes"]) != OUTCOMES:
            raise ValueError("Initial checkpoint uses a different event schema")
        return model
    return JevSnakeModel.load_base(config).float().cuda()


def run_online(config, output):
    validate_online_config(config)
    require_slurm()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(f"Use a fresh run directory: {output}")
    model = _model(config)
    tokenizer = load_tokenizer(config["base_model"])
    checkpoint(output / "initial.pt", model, config, kind="snake_online_initial")
    metadata = {
        "mode": "strict_online_snake",
        "event_schema": "snake-first-food-or-collision-v1",
        "training_labels": "one observed event from the executed source trajectory",
        "counterfactual_training_rollouts": False,
        "seed": config["seed"],
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "git_commit": os.environ["JEV_GIT_COMMIT"],
        "initial_sha256": digest(output / "initial.pt"),
        "config": config,
    }
    save_json(output / "resolved_config.json", metadata)
    incumbent = UniformPolicy()
    global_step = 0
    generation = 0
    summaries = []
    microbatch = config["microbatch"]
    with (output / "training.jsonl").open("w") as training_log:
        while (
            generation < config["max_policy_generations"]
            and global_step + config["min_optimizer_steps_per_policy"]
            <= config["max_total_optimizer_steps"]
        ):
            collector = OnlineCollector(incumbent, config, generation, "train")
            validation_collector = OnlineCollector(
                incumbent, config, generation, "validation"
            )
            validation = validation_collector.collect(config["validation_events"])
            folder = output / f"generation_{generation:03d}"
            folder.mkdir(parents=True, exist_ok=False)
            write_jsonl(
                folder / "validation.jsonl",
                validation,
            )
            target_policy_id = collector.policy_id
            save_json(
                folder / "target_policy.json",
                {
                    "base": incumbent.spec,
                    "behavior_policy_id": target_policy_id,
                    "epsilon": config["behavior_epsilon"],
                },
            )
            optimizer = torch.optim.AdamW(
                [
                    {"params": model.backbone.parameters(), "lr": config["backbone_lr"]},
                    {"params": model.head.parameters(), "lr": config["head_lr"]},
                ],
                weight_decay=config["weight_decay"],
                foreach=False,
            )

            def schedule(step):
                if step < config["warmup_steps"]:
                    return (step + 1) / max(1, config["warmup_steps"])
                return max(
                    0.0,
                    (config["max_optimizer_steps_per_policy"] - step)
                    / max(
                        1,
                        config["max_optimizer_steps_per_policy"]
                        - config["warmup_steps"],
                    ),
                )

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            policy_step = 0
            accepted = False
            candidate = None
            latest_decision = None
            started = time.monotonic()
            events_file = (folder / "events.jsonl").open("w")
            try:
                while (
                    policy_step < config["max_optimizer_steps_per_policy"]
                    and global_step < config["max_total_optimizer_steps"]
                ):
                    collection_started = time.monotonic()
                    rows = collector.collect(config["effective_batch"])
                    collection_seconds = time.monotonic() - collection_started
                    for row in rows:
                        events_file.write(json.dumps(row, allow_nan=False) + "\n")
                    events_file.flush()
                    record, microbatch = optimization_step(
                        model,
                        tokenizer,
                        rows,
                        optimizer,
                        scheduler,
                        config,
                        microbatch,
                    )
                    global_step += 1
                    policy_step += 1
                    record.update(
                        step=global_step,
                        generation=generation,
                        policy_step=policy_step,
                        collection_seconds=collection_seconds,
                        environment=dict(collector.counters),
                        target_policy_id=target_policy_id,
                    )
                    at_limit = (
                        policy_step == config["max_optimizer_steps_per_policy"]
                        or global_step == config["max_total_optimizer_steps"]
                    )
                    if policy_step % config["eval_every"] == 0 or at_limit:
                        probabilities = predict(
                            model,
                            tokenizer,
                            validation,
                            config["inference_questions"],
                            config["max_length"],
                            config["inference_precision"],
                        )
                        metrics = observed_metrics(probabilities, validation)
                        save_json(folder / "dev" / f"step_{policy_step:06d}.json", metrics)
                        record["dev"] = metrics
                    training_log.write(json.dumps(record, allow_nan=False) + "\n")
                    training_log.flush()
                    print(json.dumps(record, allow_nan=False), flush=True)
                    eligible = policy_step >= config["min_optimizer_steps_per_policy"]
                    scheduled = (
                        policy_step == config["min_optimizer_steps_per_policy"]
                        or (
                            policy_step > config["min_optimizer_steps_per_policy"]
                            and (
                                policy_step - config["min_optimizer_steps_per_policy"]
                            )
                            % config["promotion_interval_steps"]
                            == 0
                        )
                    )
                    if not eligible or not (scheduled or at_limit):
                        continue
                    checkpoint(
                        folder / "checkpoint.pt",
                        model,
                        config | {"microbatch": microbatch},
                        kind="snake_online_generation",
                        generation=generation,
                        policy_step=policy_step,
                        global_step=global_step,
                        target_policy={
                            "base": incumbent.spec,
                            "behavior_policy_id": target_policy_id,
                            "epsilon": config["behavior_epsilon"],
                        },
                        optimizer=optimizer.state_dict()
                        if config["save_optimizer"]
                        else None,
                        scheduler=scheduler.state_dict(),
                        torch_rng=torch.get_rng_state(),
                        cuda_rng=torch.cuda.get_rng_state(),
                    )
                    candidate = snapshot_candidate(
                        model,
                        tokenizer,
                        folder / "checkpoint.pt",
                        config,
                        target_policy_id,
                    )
                    seeds = [
                        stream_seed(
                            config["seed"], generation, policy_step, "promotion", index
                        )
                        for index in range(config["promotion_games"])
                    ]
                    old_games = play_seeded(incumbent, seeds, config)
                    new_games = play_seeded(candidate, seeds, config)
                    latest_decision = promotion_decision(
                        old_games,
                        new_games,
                        config["promotion_min_food_gain"],
                        config["promotion_z"],
                    )
                    promotion = {
                        "generation": generation,
                        "policy_step": policy_step,
                        "target_policy_id": target_policy_id,
                        "decision": latest_decision,
                        "incumbent": {
                            key: value
                            for key, value in old_games.items()
                            if key != "records"
                        },
                        "candidate": {
                            key: value
                            for key, value in new_games.items()
                            if key != "records"
                        },
                    }
                    save_json(folder / f"promotion_step_{policy_step:06d}.json", promotion)
                    save_json(folder / "promotion.json", promotion)
                    print(json.dumps({"promotion": promotion}, allow_nan=False), flush=True)
                    accepted = latest_decision["accepted"]
                    if accepted or at_limit:
                        break
                    del candidate
                    candidate = None
            finally:
                events_file.close()
            if candidate is None:
                raise RuntimeError("Generation ended without a candidate checkpoint")
            probabilities = predict(
                model,
                tokenizer,
                validation,
                config["inference_questions"],
                config["max_length"],
                config["inference_precision"],
            )
            metrics = observed_metrics(probabilities, validation)
            questions = action_questions(validation, config)
            references = mc_reference(
                questions,
                incumbent,
                config["behavior_epsilon"],
                config["mc_rollouts"],
                stream_seed(config["seed"], generation, "mc"),
            )
            write_jsonl(folder / "mc_reference.jsonl", references)
            mc_predictions = predict(
                model,
                tokenizer,
                references,
                config["inference_questions"],
                config["max_length"],
                config["inference_precision"],
            )
            metrics.update(mc_metrics(mc_predictions, references))
            metrics["action_ranking"] = action_ranking_metrics(
                mc_predictions, references
            )
            test_seeds = [
                stream_seed(config["seed"], generation, "test", index)
                for index in range(config["test_games"])
            ]
            incumbent_test = play_seeded(incumbent, test_seeds, config)
            candidate_test = play_seeded(candidate, test_seeds, config)
            summary = {
                "generation": generation,
                "objective": config["objective"],
                "seed": config["seed"],
                "target_policy_id": target_policy_id,
                "accepted": accepted,
                "status": "promoted" if accepted else "policy_stalled",
                "optimizer_steps": policy_step,
                "probability": metrics,
                "promotion": latest_decision,
                "candidate_test": {
                    key: value
                    for key, value in candidate_test.items()
                    if key != "records"
                },
                "incumbent_test": {
                    key: value
                    for key, value in incumbent_test.items()
                    if key != "records"
                },
                "environment": dict(collector.counters),
                "training_stats": {
                    "training_seconds": time.monotonic() - started,
                    "actual_microbatch": microbatch,
                    "peak_gpu_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                    "peak_gpu_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
                },
            }
            summaries.append(summary)
            save_json(output / "generations.json", summaries)
            print(json.dumps({"generation_complete": summary}, allow_nan=False), flush=True)
            del optimizer, scheduler, collector, validation_collector
            if not accepted:
                break
            incumbent = candidate
            generation += 1
    return summaries
