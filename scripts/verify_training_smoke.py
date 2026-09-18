"""CPU audit of a completed paired-PG smoke; creates the full-run gate."""

import json
from pathlib import Path

import numpy as np
import torch

from jev2048.utils import digest, save_json

if __name__ == "__main__":
    root = Path("runs/paired_pg_smoke")
    records = [
        json.loads(x) for x in (root / "training.jsonl").read_text().splitlines()
    ]
    metadata = json.loads((root / "resolved_config.json").read_text())
    config = metadata["config"]
    assert config["objective"] == "paired_pg" and len(records) == config["steps"] >= 10
    assert all(
        np.isfinite(r["loss"])
        and np.isfinite(r["gradient_norm"])
        and r["gradient_norm"] > 0
        for r in records
    )
    predictions = json.loads((root / "test_predictions.json").read_text())
    probs = np.array([r["probabilities"] for r in predictions])
    assert (
        np.isfinite(probs).all()
        and (probs >= 0).all()
        and np.allclose(probs.sum(-1), 1, atol=1e-6)
    )
    common = torch.load(
        config["common_init"], map_location="cpu", weights_only=False, mmap=True
    )
    saved = torch.load(
        root / "checkpoint.pt", map_location="cpu", weights_only=False, mmap=True
    )
    changes = {}
    for prefix in ("backbone.", "head."):
        keys = [
            k
            for k in saved["model"]
            if k.startswith(prefix) and saved["model"][k].is_floating_point()
        ]
        changes[prefix] = sum(
            int(not torch.equal(saved["model"][k], common["model"][k])) for k in keys
        )
        assert changes[prefix] > 0
    assert not any("visual" in k or "lm_head" in k for k in saved["model"])
    assert metadata["common_init_sha256"] == digest(config["common_init"])
    report = dict(
        success=True,
        objective="paired_pg",
        steps=len(records),
        updated_tensors=changes,
        stats=json.loads((root / "training_stats.json").read_text()),
        metrics=json.loads((root / "metrics.json").read_text()),
        git_commit=metadata["git_commit"],
        common_init_sha256=metadata["common_init_sha256"],
    )
    save_json("results/training_smoke.json", report)
    print(json.dumps(report, indent=2))
