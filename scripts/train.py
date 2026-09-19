#!/usr/bin/env python3
"""Train the pure-online Snake outcome model."""

import argparse
from pathlib import Path

from jevsnake.config import load_config
from jevsnake.trainer import run_online
from jevsnake.utils import seed_all


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/online.yaml")
    parser.add_argument("--output")
    parser.add_argument("--objective", choices=("ce", "brier", "paired_pg"))
    args = parser.parse_args()
    config = load_config(args.config)
    if args.objective:
        config["objective"] = args.objective
    seed_all(config["seed"])
    output = Path(
        args.output
        or f"runs/snake_online_{config['objective']}_seed{config['seed']}"
    )
    run_online(config, output)


if __name__ == "__main__":
    main()
