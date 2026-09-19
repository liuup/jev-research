import argparse
import json
from jev2048.training_plot import plot_training_run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        default="runs/online_expected_log_tile_smoke",
    )
    parser.add_argument("--output", default=None)
    parser.add_argument("--smooth", type=int, default=25)
    parser.add_argument("--title")
    args = parser.parse_args()
    if args.smooth < 1:
        parser.error("--smooth must be positive")
    result = plot_training_run(
        args.run,
        output=args.output,
        smooth=args.smooth,
        title=args.title,
    )
    print(json.dumps(result))


if __name__ == "__main__":
    main()
