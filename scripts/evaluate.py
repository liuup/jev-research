#!/usr/bin/env python3
"""Evaluate a saved Snake checkpoint on a frozen seed cohort."""

import argparse
import json

from jevsnake.evaluation import play_seeded
from jevsnake.model import load_tokenizer
from jevsnake.policies import FrozenModelPolicy
from jevsnake.runtime import load_checkpoint, require_slurm
from jevsnake.utils import digest, save_json, stream_seed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--output", required=True)
    parser.add_argument("--games", type=int)
    args = parser.parse_args()
    require_slurm()
    model, saved = load_checkpoint(args.checkpoint)
    config = saved["config"]
    tokenizer = load_tokenizer(config["base_model"])
    spec = {
        "kind": "jev_snake_evaluation",
        "policy_id": "jev-snake-eval:" + digest(args.checkpoint),
        "checkpoint": args.checkpoint,
    }
    policy = FrozenModelPolicy(model, tokenizer, spec, config)
    count = args.games or config["test_games"]
    seeds = [stream_seed(config["seed"], "standalone-test", i) for i in range(count)]
    result = play_seeded(policy, seeds, config)
    save_json(args.output, result)
    print(json.dumps({key: value for key, value in result.items() if key != "records"}))


if __name__ == "__main__":
    main()
