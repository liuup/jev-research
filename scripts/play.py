import argparse
import hashlib
import json
import time
from pathlib import Path

from jev2048.env import Game
from jev2048.model import load_tokenizer
from jev2048.online_evaluation import play_seeded
from jev2048.policies import FrozenHybridPolicy, FrozenModelPolicy, restore_policy
from jev2048.runtime import load_checkpoint, require_slurm
from jev2048.utils import game_summary, save_json, seed_all

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--games", type=int, default=10)
    p.add_argument("--seed", type=int, default=9017)
    p.add_argument("--utility", choices=["threshold", "log_tile"], default="threshold")
    p.add_argument("--threshold", type=int, default=2048)
    p.add_argument("--policy", choices=["active", "candidate"], default="active")
    p.add_argument("--inference-precision", choices=["fp32", "bf16"], default=None)
    p.add_argument(
        "--batch-games",
        type=int,
        default=1,
        help="Games played in one batched forward; 1 plays one game at a time.",
    )
    p.add_argument(
        "--microbatch-questions",
        type=int,
        default=None,
        help="Candidate questions per forward; defaults to the checkpoint value.",
    )
    p.add_argument(
        "--controller",
        choices=["model", "hybrid"],
        default="model",
        help="model judges every step; hybrid lets the heuristic plan and the model judge near-equal moves.",
    )
    p.add_argument("--judge-delta", type=float, default=0.0)
    p.add_argument("--judge-margin", type=float, default=0.0)
    p.add_argument("--heuristic-config", default="configs/heuristic.yaml")
    a = p.parse_args()
    if a.games < 1:
        p.error("--games must be positive")
    if a.batch_games < 1:
        p.error("--batch-games must be positive")
    if a.judge_delta < 0 or a.judge_margin < 0:
        p.error("--judge-delta and --judge-margin must be nonnegative")
    if a.microbatch_questions is not None and a.microbatch_questions < 1:
        p.error("--microbatch-questions must be positive")
    online = (Path(a.run) / "active_policy.json").exists()
    if a.controller == "hybrid" and online:
        p.error("--controller hybrid requires a local checkpoint run")
    output = Path(a.run) / (
        f"controller_{a.policy}.json" if online else f"controller_{a.utility}.json"
    )
    if a.inference_precision:
        output = output.with_name(f"{output.stem}_{a.inference_precision}.json")
    if a.controller == "hybrid":
        output = output.with_name(f"{output.stem}_hybrid_d{a.judge_delta:g}.json")
    if a.batch_games > 1:
        output = output.with_name(f"{output.stem}_batch{a.batch_games}.json")
    game_log = output.with_suffix(".jsonl")
    if output.exists() or game_log.exists():
        raise FileExistsError(f"Evaluation output already exists: {output}")
    require_slurm()
    seed_all(a.seed)
    if online:
        spec_path = Path(a.run) / f"{a.policy}_policy.json"
        if not spec_path.exists():
            raise ValueError("Candidate evaluation requires a generation directory")
        spec = json.loads(spec_path.read_text())
        if spec["kind"] == "jev" and a.inference_precision:
            spec["inference_precision"] = a.inference_precision
            spec["policy_id"] = "jev-eval:" + hashlib.sha256(
                json.dumps(spec, sort_keys=True).encode()
            ).hexdigest()
        metadata_path = Path(a.run) / "resolved_config.json"
        if not metadata_path.exists():
            metadata_path = Path(a.run).parent / "resolved_config.json"
        c = json.loads(metadata_path.read_text())["config"]
        if a.microbatch_questions is not None:
            c = dict(c, inference_questions=a.microbatch_questions)
        policy = restore_policy(spec, load_tokenizer(c["base_model"]), c)
    else:
        model, saved = load_checkpoint(a.run + "/checkpoint.pt")
        c = saved["config"]
        del saved
        if a.microbatch_questions is not None:
            c = dict(c, inference_questions=a.microbatch_questions)
        spec = dict(
            kind="jev",
            policy_id=f"jev-checkpoint:{a.run}",
            utility=a.utility,
            threshold=a.threshold,
            inference_precision=a.inference_precision or "fp32",
            task=c.get("task", "terminal_max_tile"),
            continuation_policy_id=c.get("continuation_policy_id"),
        )
        tokenizer = load_tokenizer(c["base_model"])
        if a.controller == "hybrid":
            judge = dict(
                run=a.run,
                delta=a.judge_delta,
                margin=a.judge_margin,
                heuristic_config=a.heuristic_config,
                utility=a.utility,
                threshold=a.threshold,
                inference_precision=spec["inference_precision"],
                task=spec["task"],
                continuation_policy_id=spec["continuation_policy_id"],
            )
            spec.update(
                kind="jev-hybrid",
                judge_delta=a.judge_delta,
                judge_margin=a.judge_margin,
                heuristic_config=a.heuristic_config,
                policy_id="jev-hybrid:"
                + hashlib.sha256(
                    json.dumps(judge, sort_keys=True).encode()
                ).hexdigest(),
            )
            policy = FrozenHybridPolicy(model, tokenizer, spec, c)
        else:
            policy = FrozenModelPolicy(model, tokenizer, spec, c)
    print(
        json.dumps(dict(model_config=c, controller_config=vars(a)), indent=2),
        flush=True,
    )
    seeds = range(a.seed, a.seed + a.games)
    success_tile = 2048 if c.get("task") == "reach_2048" else None
    records = []
    total_seconds = 0.0
    with game_log.open("w") as log:
        if a.batch_games > 1:
            started = time.perf_counter()
            played = play_seeded(
                policy,
                seeds,
                batch_size=a.batch_games,
                game_factory=Game,
                progress=dict(phase="closed_loop", run=a.run, role=a.policy),
                success_tile=success_tile,
            )
            total_seconds = time.perf_counter() - started
            records = played["records"]
            summary = {
                k: v for k, v in played.items() if k not in ("records", "policy_id")
            }
            for record in records:
                log.write(json.dumps(record) + "\n")
            log.flush()
            print(json.dumps(dict(games=len(records), batch_games=a.batch_games)), flush=True)
        else:
            games = []
            for seed in seeds:
                start = time.perf_counter()
                g = Game(seed)
                while not g.terminal and not (
                    c.get("task") == "reach_2048" and g.max_tile >= 2048
                ):
                    g.step(policy.choose(g))
                elapsed = time.perf_counter() - start
                total_seconds += elapsed
                games.append(g)
                record = dict(
                    seed=seed,
                    score=g.score,
                    max_tile=g.max_tile,
                    steps=g.steps,
                    seconds=elapsed,
                    seconds_per_step=elapsed / g.steps,
                )
                records.append(record)
                log.write(json.dumps(record) + "\n")
                log.flush()
                print(json.dumps(record), flush=True)
            summary = game_summary(games)
    judge_report = policy.summary() if hasattr(policy, "summary") else {}
    report = summary | dict(
        seed=a.seed,
        batch_games=a.batch_games,
        controller=a.controller,
        judge_delta=a.judge_delta if a.controller == "hybrid" else None,
        judge_margin=a.judge_margin if a.controller == "hybrid" else None,
        utility=spec.get("utility", "heuristic") if online else a.utility,
        threshold=spec.get("threshold") if online else a.threshold,
        policy_id=policy.policy_id,
        inference_precision=(
            spec.get("inference_precision")
            if online and spec["kind"] == "jev"
            else None
            if online
            else a.inference_precision or "fp32"
        ),
        total_seconds=total_seconds,
        mean_game_seconds=total_seconds / len(records),
        seconds_per_step=total_seconds / sum(r["steps"] for r in records),
        interpretation="Closed-loop policy performance; distinct from the checkpoint's frozen-policy probability target",
    ) | judge_report
    save_json(output, report)
    print(report)
