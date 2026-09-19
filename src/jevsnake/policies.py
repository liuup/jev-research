"""Immutable policy interfaces for online policy iteration."""

import copy
import hashlib
import json

from .controller import OutcomeController
from .utils import digest


def _identity(spec):
    return hashlib.sha256(
        json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


class UniformPolicy:
    def __init__(self):
        self.spec = {"kind": "uniform_non_reverse", "version": 1}
        self.policy_id = "uniform:" + _identity(self.spec)
        self.spec["policy_id"] = self.policy_id

    def action_probabilities(self, games):
        result = []
        for game in games:
            actions = game.legal_actions
            if not actions:
                raise ValueError("Cannot act in a terminal game")
            result.append({action: 1.0 / len(actions) for action in actions})
        return result


class FrozenModelPolicy:
    def __init__(self, model, tokenizer, spec, config):
        self.spec = copy.deepcopy(spec)
        self.policy_id = self.spec["policy_id"]
        self.model = model.eval().requires_grad_(False)
        self.controller = OutcomeController(self.model, tokenizer, config)

    def action_probabilities(self, games):
        return self.controller.action_probabilities(games)


def snapshot_candidate(model, tokenizer, checkpoint_path, config, target_policy_id):
    checksum = digest(checkpoint_path)
    spec = {
        "kind": "jev_snake",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checksum,
        "prediction_policy_id": target_policy_id,
        "event_horizon": config["event_horizon"],
        "event_utilities": config["utilities"],
        "inference_precision": config["inference_precision"],
    }
    spec["policy_id"] = "jev-snake:" + _identity(spec)
    return FrozenModelPolicy(copy.deepcopy(model), tokenizer, spec, config)


def behavior_policy_id(policy, epsilon):
    spec = {
        "kind": "epsilon_mixture",
        "base_policy_id": policy.policy_id,
        "epsilon": epsilon,
        "exploration": "uniform_non_reverse",
    }
    return "behavior:" + _identity(spec)
