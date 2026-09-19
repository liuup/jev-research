"""Final evaluation of a trained 24 game run on the test split."""

import argparse
import json

from jevmath.trainer import evaluate_test

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    print(json.dumps(evaluate_test(args.run), indent=2, default=str))
