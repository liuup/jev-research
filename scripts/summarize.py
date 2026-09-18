"""Build separate probability and controller tables; never invent missing results."""

import argparse
import csv
import json
from pathlib import Path

from jev2048.utils import save_json

FIELDS = [
    "objective",
    "seed",
    "top1_accuracy",
    "observed_nll",
    "observed_brier",
    "mc_squared_l2",
    "mc_mae",
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


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--runs",
        nargs="+",
        default=["runs/ce_seed17", "runs/brier_seed17", "runs/paired_pg_seed17"],
    )
    p.add_argument("--output", default="results/comparison")
    a = p.parse_args()
    rows = []
    demos = []
    for run in a.runs:
        root = Path(run)
        if not (root / "metrics.json").exists():
            continue
        c = json.loads((root / "resolved_config.json").read_text())["config"]
        r = (
            dict(objective=c["objective"], seed=c["seed"])
            | json.loads((root / "metrics.json").read_text())
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
            "utility",
            "games",
            "seed",
            "mean_score",
            "median_score",
            "max_tile_distribution",
            "reach_512",
            "reach_1024",
            "reach_2048",
        ],
        a.output + "_closed_loop",
    )
