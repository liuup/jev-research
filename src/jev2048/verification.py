"""CPU audit of a completed two-generation online GPU smoke."""

import json
import math
from pathlib import Path

import torch

from jev2048.online import validate_events


def audit(run, require_gpu=False):
    root = Path(run)
    metadata = json.loads((root / "resolved_config.json").read_text())
    if require_gpu:
        assert metadata["device_type"] == "cuda" and metadata["slurm_job_id"]
    c = metadata["config"]
    assert c["mode"] == "online" and c["objective"] == "paired_pg"
    generations = json.loads((root / "generations.json").read_text())
    assert len(generations) == c["generations"] >= 2
    rows = [
        json.loads(line) for line in (root / "training.jsonl").read_text().splitlines()
    ]
    assert len(rows) == c["generations"] * c["steps_per_generation"]
    assert all(math.isfinite(row["loss"]) and row["gradient_norm"] > 0 for row in rows)
    assert all("peak_gpu_gib" not in row for row in rows)
    initial = torch.load(
        root / "initial.pt", map_location="cpu", weights_only=False, mmap=True
    )
    final = torch.load(
        root / f"generation_{len(generations) - 1:03d}" / "checkpoint.pt",
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    changed = {}
    for prefix in ("backbone.", "head."):
        changed[prefix] = sum(
            not torch.equal(value, initial["model"][key])
            for key, value in final["model"].items()
            if key.startswith(prefix)
        )
        assert changed[prefix] > 0
    previous = metadata["initial_policy"]["policy_id"]
    for generation in generations:
        number = generation["generation"]
        assert generation["target_policy_id"] == previous
        events = [
            json.loads(line)
            for line in (root / f"generation_{number:03d}" / "events.jsonl")
            .read_text()
            .splitlines()
        ]
        validate_events(events, previous, number)
        assert len(events) == c["steps_per_generation"] * c["effective_batch"]
        previous = generation["active_policy_id"]
    return dict(
        success=True,
        mode="online",
        generations=len(generations),
        optimizer_steps=len(rows),
        changed_tensors=changed,
        run=str(root),
        device_type=metadata["device_type"],
    )
