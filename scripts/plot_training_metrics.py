"""Plot observed NLL and vector Brier curves from an online training run."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from jev2048.serialization import OUTCOMES


def read_records(path):
    records = []
    lines = Path(path).read_text().splitlines()
    for index, line in enumerate(lines):
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            if index != len(lines) - 1:
                raise
    if not records:
        raise ValueError(f"No complete records in {path}")
    return records


def moving_average(values, window):
    values = np.asarray(values, dtype=float)
    if window <= 1:
        return values
    result = np.full(len(values), np.nan)
    if len(values) >= window:
        result[window - 1 :] = np.convolve(
            values, np.ones(window) / window, mode="valid"
        )
    return result


def mean_log_tile_bias(row, split):
    metrics = row[split]
    if "expected_log_tile_mean_bias" in metrics:
        return metrics["expected_log_tile_mean_bias"]
    if split == "dev":
        return (
            metrics["predicted_expected_log_tile"]
            - metrics["observed_mean_log_tile"]
        )
    utilities = np.arange(7, 14, dtype=float)
    predicted = np.array(
        [row["mean_predicted_distribution"][outcome] for outcome in OUTCOMES]
    )
    counts = np.array(
        [row["observed_outcome_counts"][outcome] for outcome in OUTCOMES]
    )
    return float(predicted @ utilities - counts @ utilities / counts.sum())


def metric(row, split, key):
    if key == "expected_log_tile_mean_bias":
        return mean_log_tile_bias(row, split)
    return row[split][key]


def plot_metrics(records, output, smooth):
    steps = np.array([row["step"] for row in records])
    generations = np.array([row["generation"] for row in records])
    definitions = (
        ("observed_nll", "Observed NLL"),
        ("observed_brier", "Vector Brier score"),
        ("expected_log_tile_mean_bias", "Mean expected log-tile bias"),
    )
    figure, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    for axis, (key, label) in zip(axes, definitions):
        train = np.array([metric(row, "train", key) for row in records])
        axis.plot(steps, train, color="#4C78A8", alpha=0.18, linewidth=0.8, label="train batch")
        axis.plot(
            steps,
            moving_average(train, smooth),
            color="#4C78A8",
            linewidth=2,
            label=f"train rolling mean ({smooth})",
        )
        dev = [
            (row["step"], metric(row, "dev", key))
            for row in records
            if "dev" in row
        ]
        if dev:
            axis.plot(
                [item[0] for item in dev],
                [item[1] for item in dev],
                color="#F58518",
                marker="o",
                markersize=3.5,
                linewidth=1.5,
                label="dev",
            )
        axis.set_ylabel(label)
        if key == "expected_log_tile_mean_bias":
            axis.axhline(0, color="black", linestyle="--", alpha=0.5)
        axis.grid(alpha=0.25)
        axis.legend()
    for index in np.flatnonzero(generations[1:] != generations[:-1]) + 1:
        for axis in axes:
            axis.axvline(steps[index], color="black", linestyle="--", alpha=0.35)
    axes[-1].set_xlabel("Optimizer step")
    figure.suptitle(
        f"Online paired-PG probability metrics ({len(records):,} completed steps)"
    )
    figure.tight_layout()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        default="runs/online_expected_log_tile_smoke",
    )
    parser.add_argument("--output", default=None)
    parser.add_argument("--smooth", type=int, default=25)
    args = parser.parse_args()
    if args.smooth < 1:
        parser.error("--smooth must be positive")
    run = Path(args.run)
    output = Path(args.output) if args.output else Path("results/plots") / f"{run.name}_nll_brier.png"
    records = read_records(run / "training.jsonl")
    plot_metrics(records, output, args.smooth)
    print(json.dumps(dict(output=str(output), steps=len(records), smooth=args.smooth)))


if __name__ == "__main__":
    main()
