#!/usr/bin/env python3
"""Validate full and smoke configurations without loading a model."""

from jevsnake.config import load_config
from jevsnake.trainer import validate_online_config


for path in ("configs/online.yaml", "configs/online_smoke.yaml"):
    config = load_config(path)
    validate_online_config(config)
    print(f"valid: {path}")
