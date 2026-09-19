"""Versioned event semantics shared by training and control."""

import numpy as np

OUTCOMES = ("collision", "timeout", "food")
DESCRIPTIONS = {
    "collision": "A wall or body collision occurs before the next food is collected.",
    "timeout": "Neither a collision nor food collection occurs within the stated horizon.",
    "food": "The next food is collected before a wall or body collision occurs.",
}
UTILITIES = np.array([-1.0, 0.0, 1.0], dtype=np.float64)
EVENT_SCHEMA = "snake-first-food-or-collision-v1"


def validate_event_config(config):
    if tuple(config["outcomes"]) != OUTCOMES:
        raise ValueError(f"outcomes must be exactly {list(OUTCOMES)}")
    utilities = np.asarray(config["utilities"], dtype=float)
    if utilities.shape != (len(OUTCOMES),) or not np.array_equal(
        utilities, UTILITIES
    ):
        raise ValueError(f"utilities must be exactly {UTILITIES.tolist()}")
    if not isinstance(config["event_horizon"], int) or config["event_horizon"] < 1:
        raise ValueError("event_horizon must be a positive integer")
