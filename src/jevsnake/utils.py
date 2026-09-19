"""Small deterministic helpers."""

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch


def stream_seed(seed, *parts):
    message = ":".join(map(str, (seed, *parts)))
    return int.from_bytes(hashlib.sha256(message.encode()).digest()[:8], "big") % (
        2**63 - 1
    )


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
