import argparse
import hashlib
import json
import time
from pathlib import Path

from jev2048.controller import JevController
from jev2048.env import Game
from jev2048.model import load_tokenizer
from jev2048.policies import restore_policy
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
    a = p.parse_args()
    if a.games < 1:
        p.error("--games must be positive")
    online = (Path(a.run) / "active_policy.json").exists()
    output = Path(a.run) / (
        f"controller_{a.policy}.json" if online else f"controller_{a.utility}.json"
    )
    if a.inference_precision:
        output = output.with_name(f"{output.stem}_{a.inference_precision}.json")
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
        controller = restore_policy(spec, load_tokenizer(c["base_model"]), c)
    else:
        model, saved = load_checkpoint(a.run + "/checkpoint.pt")
        c = saved["config"]
        del saved
        controller = JevController(
            model,
            load_tokenizer(c["base_model"]),
            a.utility,
            a.threshold,
            c["max_length"],
            inference_precision=a.inference_precision or "fp32",
            task=c.get("task", "terminal_max_tile"),
            continuation_policy_id=c.get("continuation_policy_id"),
        )
    print(
        json.dumps(dict(model_config=c, controller_config=vars(a)), indent=2),
        flush=True,
    )
    games = []
    total_seconds = 0.0
    with game_log.open("w") as log:
        for seed in range(a.seed, a.seed + a.games):
            start = time.perf_counter()
            g = Game(seed)
            while not g.terminal and not (c.get("task") == "reach_2048" and g.max_tile >= 2048):
                g.step(controller.choose(g))
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
            log.write(json.dumps(record) + "\n")
            log.flush()
            print(json.dumps(record), flush=True)
    report = game_summary(games) | dict(
        seed=a.seed,
        utility=spec.get("utility", "heuristic") if online else a.utility,
        threshold=spec.get("threshold") if online else a.threshold,
        policy_id=spec["policy_id"] if online else "legacy_checkpoint",
        inference_precision=spec.get("inference_precision", "bf16") if online and spec["kind"] == "jev" else (a.inference_precision or "fp32") if not online else None,
        total_seconds=total_seconds,
        mean_game_seconds=total_seconds / len(games),
        seconds_per_step=total_seconds / sum(g.steps for g in games),
        interpretation="Closed-loop policy performance; distinct from the checkpoint's frozen-policy probability target",
    )
    save_json(output, report)
    print(report)
