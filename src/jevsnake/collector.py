"""Strict online trajectory collector with delayed first-event labels.

Only the action actually executed in a source trajectory receives a training
label.  No counterfactual action clone or evaluation-only MC distribution enters
the training batch.
"""

import random
from collections import deque

from .env import SnakeGame
from .events import OUTCOMES
from .policies import behavior_policy_id
from .utils import stream_seed


def _sample(mapping, rng):
    if not mapping or any(value < 0 for value in mapping.values()):
        raise ValueError("Invalid action distribution")
    total = sum(mapping.values())
    if not 0.999999 <= total <= 1.000001:
        raise ValueError("Action probabilities must sum to one")
    draw, cumulative = rng.random(), 0.0
    for action, probability in mapping.items():
        cumulative += probability / total
        if draw < cumulative:
            return action
    return next(reversed(mapping))


def epsilon_mix(mapping, epsilon):
    if not 0 <= epsilon <= 1:
        raise ValueError("epsilon must be in [0,1]")
    uniform = 1.0 / len(mapping)
    return {
        action: (1 - epsilon) * probability + epsilon * uniform
        for action, probability in mapping.items()
    }


def validate_events(rows, policy_id, generation, stream="train"):
    if not rows:
        raise ValueError("Empty event batch")
    for row in rows:
        if (
            row["policy_id"] != policy_id
            or row["generation"] != generation
            or row["stream"] != stream
        ):
            raise ValueError("Event belongs to a different policy, generation, or stream")
        if row["observed_outcome"] not in OUTCOMES or "q" in row:
            raise ValueError("Training requires one observed event and never an MC target")


class OnlineCollector:
    def __init__(self, policy, config, generation, stream="train"):
        self.policy = policy
        self.config = config
        self.generation = generation
        self.stream = stream
        self.policy_id = behavior_policy_id(policy, config["behavior_epsilon"])
        self.pending = [[] for _ in range(config["source_envs"])]
        self.completed = deque()
        self.games = [None] * config["source_envs"]
        self.episode_ids = [None] * config["source_envs"]
        self.rngs = [None] * config["source_envs"]
        self.serial = 0
        self.counters = {
            "episodes_started": 0,
            "behavior_steps": 0,
            "events_started": 0,
            "events_completed": 0,
            "food_events": 0,
            "collision_events": 0,
            "timeout_events": 0,
        }
        for index in range(config["source_envs"]):
            self._reset(index)

    def _reset(self, index):
        serial = self.serial
        self.serial += 1
        seed = stream_seed(
            self.config["seed"], self.generation, self.stream, "episode", serial
        )
        self.games[index] = SnakeGame(
            seed, self.config["board_size"], self.config["initial_length"]
        )
        self.episode_ids[index] = (
            f"{self.stream}:g{self.generation}:episode{serial}:seed{seed}"
        )
        self.rngs[index] = random.Random(
            stream_seed(self.config["seed"], self.episode_ids[index], "behavior")
        )
        self.pending[index].clear()
        self.counters["episodes_started"] += 1

    def _finish(self, index, outcome, event_steps, pending):
        row = pending["row"] | {
            "observed_outcome": outcome,
            "event_steps": event_steps,
        }
        self.completed.append(row)
        self.counters["events_completed"] += 1
        self.counters[f"{outcome}_events"] += 1

    def advance(self):
        base = self.policy.action_probabilities(self.games)
        if len(base) != len(self.games):
            raise ValueError("Policy returned the wrong batch size")
        for index, (game, distribution) in enumerate(zip(self.games, base)):
            if set(distribution) != set(game.legal_actions):
                raise ValueError("Policy action set does not match the environment")
            action = _sample(
                epsilon_mix(distribution, self.config["behavior_epsilon"]),
                self.rngs[index],
            )
            if game.steps % self.config["select_every"] == 0:
                state_id = f"{self.episode_ids[index]}:step{game.steps}"
                self.pending[index].append(
                    {
                        "start_step": game.steps,
                        "row": game.public_state()
                        | {
                            "state_id": state_id,
                            "source_episode": self.episode_ids[index],
                            "action": action,
                            "event_horizon": self.config["event_horizon"],
                            "policy_id": self.policy_id,
                            "base_policy_id": self.policy.policy_id,
                            "generation": self.generation,
                            "stream": self.stream,
                        },
                    }
                )
                self.counters["events_started"] += 1
            transition = game.step(action)
            self.counters["behavior_steps"] += 1
            survivors = []
            for pending in self.pending[index]:
                age = game.steps - pending["start_step"]
                if transition["ate_food"]:
                    self._finish(index, "food", age, pending)
                elif transition["collision"]:
                    self._finish(index, "collision", age, pending)
                elif age >= self.config["event_horizon"]:
                    self._finish(index, "timeout", age, pending)
                else:
                    survivors.append(pending)
            self.pending[index] = survivors
            if game.terminal:
                if self.pending[index]:
                    raise RuntimeError("Terminal transition left unresolved events")
                self._reset(index)

    def collect(self, count):
        if not isinstance(count, int) or count < 1:
            raise ValueError("count must be positive")
        while len(self.completed) < count:
            self.advance()
        rows = [self.completed.popleft() for _ in range(count)]
        validate_events(rows, self.policy_id, self.generation, self.stream)
        return rows
