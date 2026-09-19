#!/usr/bin/env python3
"""One real Qwen forward/backward smoke check on the allocated GPU."""

import json
from pathlib import Path

import torch

from jevsnake.batching import collate
from jevsnake.config import load_config
from jevsnake.env import SnakeGame
from jevsnake.model import JevSnakeModel, load_tokenizer
from jevsnake.objectives import loss
from jevsnake.runtime import require_slurm
from jevsnake.utils import save_json, seed_all


def main():
    require_slurm()
    config = load_config("configs/online_smoke.yaml")
    seed_all(config["seed"])
    model = JevSnakeModel.load_base(config).float().cuda()
    tokenizer = load_tokenizer(config["base_model"])
    game = SnakeGame(config["seed"], config["board_size"], config["initial_length"])
    rows = []
    outcomes = ("timeout", "food", "collision")
    for index, action in enumerate(game.legal_actions):
        rows.append(
            game.public_state()
            | {
                "state_id": f"smoke:{index}",
                "action": action,
                "event_horizon": config["event_horizon"],
                "observed_outcome": outcomes[index],
            }
        )
    batch = collate(rows, tokenizer, config["max_length"])
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logp = model(batch)
        objective = loss(
            logp,
            batch["targets"].cuda(),
            "paired_pg",
            config["reward_samples"],
            config["baseline"],
        )
    objective.backward()
    result = {
        "rows": len(rows),
        "shape": list(logp.shape),
        "loss": float(objective.detach()),
        "probability_sums": logp.detach().exp().sum(-1).cpu().tolist(),
        "gpu": torch.cuda.get_device_name(0),
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
    }
    if not torch.isfinite(objective) or not all(
        abs(total - 1) < 1e-5 for total in result["probability_sums"]
    ):
        raise RuntimeError("Smoke test produced invalid probabilities or loss")
    save_json("results/snake_gpu_smoke.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
