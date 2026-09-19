"""File, logging and reproducibility helpers."""

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n")


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def rate(numerator, denominator):
    return float(numerator) / denominator if denominator else None


def stream_seed(seed, *parts):
    """Deterministic 63-bit seed for one stream of a run."""
    message = ":".join(map(str, (seed, *parts)))
    return int.from_bytes(hashlib.sha256(message.encode()).digest()[:8], "big") % (
        2**63 - 1
    )
