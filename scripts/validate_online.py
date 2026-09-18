"""Read-only config preflight. Never allocates a GPU or submits jobs."""

import argparse
import json
from pathlib import Path

from jev2048.config import load_config
from jev2048.online_trainer import validate_online_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_paired_pg.yaml")
    args = parser.parse_args()
    config = load_config("configs/model.yaml") | load_config(args.config)
    validate_online_config(config)
    for path in (
        Path(config["base_model"]) / "config.json",
        Path(config["heuristic_config"]),
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    print(
        json.dumps(
            dict(
                success=True,
                mode=config["mode"],
                objective=config["objective"],
                generations=config["generations"],
                steps_per_generation=config["steps_per_generation"],
                requires_offline_dataset=False,
                submits_jobs=False,
            ),
            indent=2,
        )
    )
