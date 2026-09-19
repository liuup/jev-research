"""YAML configuration loading with explicit inheritance."""

from pathlib import Path

import yaml


def load_config(path, seen=None):
    path = Path(path).resolve()
    seen = set() if seen is None else set(seen)
    if path in seen:
        raise ValueError("Configuration inheritance cycle")
    seen.add(path)
    with path.open() as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a mapping")
    parent = config.pop("inherit", None)
    return (load_config(path.parent / parent, seen) if parent else {}) | config
