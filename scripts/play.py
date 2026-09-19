#!/usr/bin/env python3
"""Print one deterministic greedy Snake episode with model event probabilities."""

import argparse
import json

from jevsnake.controller import OutcomeController, expected_utility
from jevsnake.env import SnakeGame
from jevsnake.model import load_tokenizer
from jevsnake.runtime import load_checkpoint, require_slurm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--seed", type=int, default=25)
    parser.add_argument("--max-steps", type=int, default=256)
    args = parser.parse_args()
    require_slurm()
    model, saved = load_checkpoint(args.checkpoint)
    config = saved["config"]
    tokenizer = load_tokenizer(config["base_model"])
    controller = OutcomeController(model, tokenizer, config)
    game = SnakeGame(args.seed, config["board_size"], config["initial_length"])
    while not game.terminal and game.steps < args.max_steps:
        distributions = controller.event_probabilities(game)
        values = {
            action: float(expected_utility([probabilities])[0])
            for action, probabilities in distributions.items()
        }
        action = max(values, key=values.get)
        before = game.public_state()
        transition = game.step(action)
        print(
            json.dumps(
                {
                    "state": before,
                    "event_probabilities": distributions,
                    "expected_utilities": values,
                    "action": action,
                    "transition": transition,
                },
                allow_nan=False,
            ),
            flush=True,
        )
    print(
        json.dumps(
            {
                "final": game.public_state(),
                "outcome": game.outcome or "horizon_survived",
            }
        )
    )


if __name__ == "__main__":
    main()
