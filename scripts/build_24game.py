"""Generate the 24 game puzzle files and their manifest."""

import argparse
import json

from jevmath.puzzles import write_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/24game")
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    print(json.dumps(write_dataset(args.root, seed=args.seed), indent=2))
