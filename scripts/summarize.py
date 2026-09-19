"""Build separate probability and controller tables; never invent missing results."""

import argparse
import csv
import json
from pathlib import Path

from jev2048.utils import save_json

FIELDS = [
    "objective",
    "seed",
    "generation",
    "target_policy_id",
    "top1_accuracy",
    "observed_nll",
    "observed_brier",
    "binary_brier",
    "predicted_success_probability",
    "observed_success_rate",
    "predicted_expected_log_tile",
    "observed_mean_log_tile",
    "expected_log_tile_mean_bias",
    "expected_log_tile_mean_absolute_bias",
    "mc_squared_l2",
    "mc_mae",
    "mc_expected_log_tile_mae",
    "ece_ge1024",
    "ece_ge2048",
    "peak_gpu_gib",
    "training_seconds",
]


def table(rows, fields, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_json(path.with_suffix(".json"), rows)
    with path.with_suffix(".csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    def cell(value):
        return f"{value:.6g}" if isinstance(value, float) else str(value)

    path.with_suffix(".md").write_text(
        "| "
        + " | ".join(fields)
        + " |\n| "
        + " | ".join(["---"] * len(fields))
        + " |\n"
        + "".join(
            "| " + " | ".join(cell(r.get(k, "pending")) for k in fields) + " |\n"
            for r in rows
        )
    )


def normalize_probability_metrics(metrics):
    metrics = dict(metrics)
    if (
        "expected_log_tile_mean_bias" not in metrics
        and metrics.get("predicted_expected_log_tile") is not None
        and metrics.get("observed_mean_log_tile") is not None
    ):
        bias = (
            metrics["predicted_expected_log_tile"]
            - metrics["observed_mean_log_tile"]
        )
        metrics["expected_log_tile_mean_bias"] = bias
        metrics["expected_log_tile_mean_absolute_bias"] = abs(bias)
    return metrics


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--runs",
        nargs="+",
        default=[
            "runs/offline_ce_seed17",
            "runs/offline_brier_seed17",
            "runs/offline_paired_pg_seed17",
        ],
    )
    p.add_argument("--output", default="results/comparison")
    a = p.parse_args()
    rows = []
    demos = []
    for run in a.runs:
        root = Path(run)
        if (root / "generations.json").exists():
            c = json.loads((root / "resolved_config.json").read_text())["config"]
            for generation in json.loads((root / "generations.json").read_text()):
                r = {
                    key: generation[key]
                    for key in ("objective", "seed", "generation", "target_policy_id")
                }
                r.update(normalize_probability_metrics(generation["probability"]))
                r.update(generation["training_stats"])
                rows.append({key: r.get(key) for key in FIELDS})
                for role in ("incumbent", "candidate"):
                    demos.append(
                        dict(
                            objective=c["objective"],
                            generation=generation["generation"],
                            role=role,
                            utility=c["utility"],
                            accepted=generation["accepted"],
                        )
                        | generation[f"{role}_test"]
                    )
            continue
        if not (root / "metrics.json").exists():
            continue
        c = json.loads((root / "resolved_config.json").read_text())["config"]
        r = (
            dict(objective=c["objective"], seed=c["seed"])
            | normalize_probability_metrics(
                json.loads((root / "metrics.json").read_text())
            )
            | json.loads((root / "training_stats.json").read_text())
        )
        rows.append({k: r.get(k) for k in FIELDS})
        for path in root.glob("controller_*.json"):
            demos.append(dict(objective=c["objective"]) | json.loads(path.read_text()))
    table(rows, FIELDS, a.output)
    table(
        demos,
        [
            "objective",
            "generation",
            "role",
            "accepted",
            "utility",
            "games",
            "seed",
            "mean_score",
            "median_score",
            "mean_log_tile",
            "median_log_tile",
            "max_tile_distribution",
            "reach_512",
            "reach_1024",
            "reach_2048",
        ],
        a.output + "_closed_loop",
    )
