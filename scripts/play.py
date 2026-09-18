import argparse
import json
import time
from pathlib import Path

from jev2048.controller import JevController
from jev2048.env import Game
from jev2048.model import load_tokenizer
from jev2048.trainer import load_checkpoint, require_slurm
from jev2048.utils import game_summary, save_json, seed_all

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--games", type=int, default=10)
    p.add_argument("--seed", type=int, default=9017)
    p.add_argument("--utility", choices=["threshold", "log_tile"], default="threshold")
    p.add_argument("--threshold", type=int, default=2048)
    a = p.parse_args()
    output = Path(a.run) / f"controller_{a.utility}.json"
    game_log = output.with_suffix(".jsonl")
    if output.exists() or game_log.exists():
        raise FileExistsError(f"Evaluation output already exists: {output}")
    require_slurm()
    seed_all(a.seed)
    model, saved = load_checkpoint(a.run + "/checkpoint.pt")
    c = saved["config"]
    del saved
    print(
        json.dumps(dict(model_config=c, controller_config=vars(a)), indent=2),
        flush=True,
    )
    controller = JevController(
        model, load_tokenizer(c["base_model"]), a.utility, a.threshold, c["max_length"]
    )
    games = []
    total_seconds = 0.0
    with game_log.open("w") as log:
        for seed in range(a.seed, a.seed + a.games):
            start = time.perf_counter()
            g = Game(seed)
            while not g.terminal:
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
        utility=a.utility,
        threshold=a.threshold,
        total_seconds=total_seconds,
        mean_game_seconds=total_seconds / len(games),
        seconds_per_step=total_seconds / sum(g.steps for g in games),
        interpretation="Closed-loop demonstration only; predictions target frozen pi0, not this controller",
    )
    save_json(output, report)
    print(report)
