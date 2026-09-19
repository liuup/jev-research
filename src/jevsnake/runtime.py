"""Slurm and checkpoint helpers."""

import os
from pathlib import Path

import torch

from .model import JevSnakeModel


def require_slurm():
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("GPU work requires a Slurm allocation")
    torch.set_num_threads(4)


def checkpoint(path, model, config, **extra):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": config, **extra}, path)


def load_checkpoint(path):
    require_slurm()
    saved = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    model = JevSnakeModel.load_base(saved["config"]).float()
    model.load_state_dict(saved["model"], strict=True)
    return model.cuda(), saved
