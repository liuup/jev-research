"""Immutable policy snapshots for online policy evaluation/improvement."""

import copy
import hashlib
import json
from pathlib import Path

from .controller import HeuristicJudgeController, JevController
from .heuristic import HeuristicPolicy
from .runtime import load_checkpoint
from .utils import digest


class FrozenHeuristic:
    def __init__(self, config="configs/heuristic.yaml"):
        self.policy = HeuristicPolicy(config)
        content = dict(
            kind="heuristic",
            config_path=str(Path(config).resolve()),
            config_sha256=digest(config),
            code_sha256=digest(Path(__file__).with_name("heuristic.py")),
        )
        self.policy_id = (
            "heuristic:"
            + hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()
        )
        self.spec = dict(content, policy_id=self.policy_id)

    def choose_many(self, games):
        return [self.policy.choose(game) for game in games]

    def choose(self, game):
        return self.policy.choose(game)


class FrozenModelPolicy:
    def __init__(self, model, tokenizer, spec, config):
        self.spec = copy.deepcopy(spec)
        self.policy_id = self.spec["policy_id"]
        self.model = model.eval().requires_grad_(False)
        self.controller = JevController(
            model,
            tokenizer,
            spec["utility"],
            spec["threshold"],
            config["max_length"],
            config["inference_questions"],
            spec.get("inference_precision", "bf16"),
            task=spec.get("task", "terminal_max_tile"),
            continuation_policy_id=spec.get("continuation_policy_id"),
        )

    def choose_many(self, games):
        return self.controller.choose_many(games)

    def choose(self, game):
        return self.controller.choose(game)


class FrozenHybridPolicy:
    """Frozen heuristic planner whose judgments come from a trained checkpoint."""

    def __init__(self, model, tokenizer, spec, config):
        self.spec = copy.deepcopy(spec)
        self.policy_id = self.spec["policy_id"]
        self.model = model.eval().requires_grad_(False)
        heuristic_config = spec.get("heuristic_config", "configs/heuristic.yaml")
        heuristic = FrozenHeuristic(heuristic_config)
        # Labels and checkpoints identify the continuation policy by the heuristic
        # config digest, which is the identity the frozen data contract uses.
        expected = spec.get("continuation_policy_id")
        if expected and expected != f"heuristic:{digest(heuristic_config)}":
            raise ValueError(
                "Judge continuation policy differs from the policy the labels used"
            )
        self.heuristic = heuristic
        self.controller = HeuristicJudgeController(
            model,
            tokenizer,
            heuristic.policy,
            spec["judge_delta"],
            spec.get("judge_margin", 0.0),
            config["max_length"],
            config["inference_questions"],
            spec.get("inference_precision", "fp32"),
            spec.get("task", "reach_2048"),
            spec.get("continuation_policy_id"),
        )

    def choose_many(self, games):
        return self.controller.choose_many(games)

    def choose(self, game):
        return self.controller.choose(game)

    def summary(self):
        return self.controller.summary()


def snapshot_candidate(model, tokenizer, checkpoint_path, config, target_policy_id):
    checksum = digest(checkpoint_path)
    spec = dict(
        kind="jev",
        policy_id="jev:" + checksum,
        checkpoint=str(Path(checkpoint_path).resolve()),
        checkpoint_sha256=checksum,
        prediction_policy_id=target_policy_id,
        utility=config["utility"],
        threshold=config["threshold"],
        inference_precision=config.get("inference_precision", "fp32"),
    )
    # Precision changes the physical action policy; include it in snapshot identity.
    spec["policy_id"] = "jev:" + hashlib.sha256(
        json.dumps(spec, sort_keys=True).encode()
    ).hexdigest()
    return FrozenModelPolicy(copy.deepcopy(model), tokenizer, spec, config)


def restore_policy(spec, tokenizer, config):
    if spec["kind"] == "heuristic":
        policy = FrozenHeuristic(spec["config_path"])
        if policy.policy_id != spec["policy_id"]:
            raise ValueError("Frozen heuristic has changed")
        return policy
    if spec["kind"] != "jev" or digest(spec["checkpoint"]) != spec["checkpoint_sha256"]:
        raise ValueError("Invalid policy snapshot")
    model, saved = load_checkpoint(spec["checkpoint"])
    if saved["target_policy"]["policy_id"] != spec["prediction_policy_id"]:
        raise ValueError("Snapshot prediction target mismatch")
    del saved
    return FrozenModelPolicy(model, tokenizer, spec, config)
