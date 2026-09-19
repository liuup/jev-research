"""Ten-step smoke run of the whole 24 game loop on CPU with the tiny model.

Checks the pieces the plan asks for before any GPU work: the generation runs end to end,
the rows carry exactly one observed outcome, the dev metrics are finite, and the
artifacts land on disk. Exits non-zero when any check fails.
"""

import argparse
import json
import shutil
from pathlib import Path

from jevmath.config import load_config
from jevmath.puzzles import verify_manifest
from jevmath.tiny import tiny_model_and_tokenizer
from jevmath.trainer import run, validate_config

CHECKS = (
    "puzzles",
    "rollouts",
    "training_rows",
    "solved_rate",
    "row_yes_rate",
    "mean_p_yes",
    "mean_action_entropy",
    "collect_seconds",
)


def smoke(root="runs/smoke", data_dir="data/24game", generations=1):
    manifest = verify_manifest(data_dir)
    config = load_config("configs/model.yaml") | load_config("configs/smoke.yaml")
    config = dict(config, data_dir=data_dir, generations=generations)
    validate_config(config)
    model, tokenizer = tiny_model_and_tokenizer()
    output = Path(root)
    if output.exists():
        shutil.rmtree(output)
    summaries = run(config, output, model=model, tokenizer=tokenizer)
    assert len(summaries) == generations, summaries
    report = dict(manifest=manifest["counts"], generations=[])
    for summary in summaries:
        rollout = summary["rollout"]
        for key in CHECKS:
            assert key in rollout, f"rollout metrics missing {key}"
        assert rollout["training_rows"] > 0, rollout
        dev = summary["dev"]
        for key in (
            "solved_rate",
            "binary_brier",
            "observed_nll",
            "ece_ge_correct",
            "mean_prediction_entropy",
            "oracle_separation",
        ):
            assert key in dev, f"dev metrics missing {key}"
        assert 0.0 <= dev["solved_rate"] <= 1.0
        assert dev["binary_brier"] >= 0.0
        report["generations"].append(dict(rollout=rollout, dev=dev))
    rows = [
        json.loads(line)
        for line in (output / "generation_000" / "rows.jsonl").read_text().splitlines()
    ]
    assert rows, "no training rows written"
    assert all(row["observed_outcome"] in ("yes", "no") for row in rows), rows[0]
    assert len({(row["state_id"], row["action_key"]) for row in rows}) == len(rows), (
        "training rows must be unique per (state, action)"
    )
    assert (output / "checkpoint.pt").exists(), "no checkpoint"
    assert (output / "summary.json").exists(), "no summary"
    report["rows"] = len(rows)
    report["checkpoint_bytes"] = (output / "checkpoint.pt").stat().st_size
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="runs/smoke")
    parser.add_argument("--data-dir", default="data/24game")
    parser.add_argument("--generations", type=int, default=1)
    args = parser.parse_args()
    print(json.dumps(smoke(args.root, args.data_dir, args.generations), indent=2, default=str))


if __name__ == "__main__":
    main()
