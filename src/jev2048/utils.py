import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2) + "\n")


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def game_summary(games):
    scores = [g.score for g in games]
    tiles = [g.max_tile for g in games]
    return dict(
        games=len(games),
        mean_score=float(np.mean(scores)),
        median_score=float(np.median(scores)),
        max_tile_distribution={str(t): tiles.count(t) for t in sorted(set(tiles))},
        **{
            f"reach_{t}": float(np.mean(np.array(tiles) >= t))
            for t in (512, 1024, 2048)
        },
    )
