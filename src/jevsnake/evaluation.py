"""Evaluation-only Monte Carlo references and paired closed-loop games."""

import math
import random

import numpy as np

from .collector import _sample, epsilon_mix
from .env import SnakeGame
from .events import OUTCOMES, UTILITIES
from .utils import stream_seed


def game_from_row(row):
    return SnakeGame.from_state(
        {
            "game": "snake",
            "seed": 0,
            "size": row["size"],
            "rng_state": 0,
            "body": row["body"],
            "direction": row["direction"],
            "food": row["food"],
            "score": row["score"],
            "steps": row["steps"],
            "terminal": False,
            "outcome": None,
        }
    )


def rollout_event(row, policy, epsilon, seed):
    game = game_from_row(row)
    rng = random.Random(seed)
    transition = game.step(row["action"])
    if transition["ate_food"]:
        return "food"
    if transition["collision"]:
        return "collision"
    for _ in range(1, row["event_horizon"]):
        distribution = policy.action_probabilities([game])[0]
        action = _sample(epsilon_mix(distribution, epsilon), rng)
        transition = game.step(action)
        if transition["ate_food"]:
            return "food"
        if transition["collision"]:
            return "collision"
    return "timeout"


def mc_reference(rows, policy, epsilon, rollouts, seed):
    if rollouts < 2:
        raise ValueError("MC reference requires at least two rollouts")
    result = []
    for row_index, row in enumerate(rows):
        counts = np.zeros(len(OUTCOMES), dtype=int)
        for repeat in range(rollouts):
            outcome = rollout_event(
                row,
                policy,
                epsilon,
                stream_seed(seed, row_index, repeat, "mc-event"),
            )
            counts[OUTCOMES.index(outcome)] += 1
        result.append(
            {
                **{key: value for key, value in row.items() if key != "observed_outcome"},
                "counts": counts.tolist(),
                "q": (counts / rollouts).tolist(),
                "rollouts": rollouts,
            }
        )
    return result


def mc_metrics(predictions, rows):
    reference = np.asarray([row["q"] for row in rows])
    midpoint = (predictions + reference) / 2

    def kl(left, right):
        return (
            left
            * (
                np.log(np.maximum(left, 1e-30))
                - np.log(np.maximum(right, 1e-30))
            )
        ).sum(-1)

    predicted_utility = predictions @ UTILITIES
    reference_utility = reference @ UTILITIES
    return {
        "mc_squared_l2": float(((predictions - reference) ** 2).sum(-1).mean()),
        "mc_mae": float(np.abs(predictions - reference).mean()),
        "mc_js": float((0.5 * kl(predictions, midpoint) + 0.5 * kl(reference, midpoint)).mean()),
        "mc_expected_utility_mae": float(
            np.abs(predicted_utility - reference_utility).mean()
        ),
        "mc_expected_utility_mse": float(
            ((predicted_utility - reference_utility) ** 2).mean()
        ),
        "mc_sampling_squared_l2_floor": float(
            np.mean(
                [
                    (1 - (np.asarray(row["q"]) ** 2).sum())
                    / (row["rollouts"] - 1)
                    for row in rows
                ]
            )
        ),
    }


def action_ranking_metrics(predictions, rows):
    groups = {}
    for prediction, row in zip(predictions, rows):
        groups.setdefault(row["state_id"], []).append((prediction, row))
    agreements, regrets, predicted_gaps = [], [], []
    for values in groups.values():
        if len(values) < 2:
            continue
        predicted = np.asarray([probability @ UTILITIES for probability, _ in values])
        reference = np.asarray([np.asarray(row["q"]) @ UTILITIES for _, row in values])
        predicted_choice = int(np.argmax(predicted))
        reference_choice = int(np.argmax(reference))
        agreements.append(predicted_choice == reference_choice)
        regrets.append(float(reference.max() - reference[predicted_choice]))
        ordered = np.sort(predicted)
        predicted_gaps.append(float(ordered[-1] - ordered[-2]))
    return {
        "states": len(agreements),
        "mc_greedy_agreement": float(np.mean(agreements)) if agreements else None,
        "mc_utility_regret": float(np.mean(regrets)) if regrets else None,
        "mean_predicted_action_gap": float(np.mean(predicted_gaps))
        if predicted_gaps
        else None,
        "caveat": "Finite Monte Carlo action rankings have sampling error",
    }


def play_seeded(policy, seeds, config):
    games = [
        SnakeGame(seed, config["board_size"], config["initial_length"])
        for seed in seeds
    ]
    rngs = [random.Random(stream_seed(seed, policy.policy_id, "game-policy")) for seed in seeds]
    active = list(range(len(games)))
    while active:
        batch = [games[index] for index in active]
        distributions = policy.action_probabilities(batch)
        next_active = []
        for index, distribution in zip(active, distributions):
            action = _sample(distribution, rngs[index])
            games[index].step(action)
            if not games[index].terminal and games[index].steps < config["episode_horizon"]:
                next_active.append(index)
        active = next_active
    records = [
        {
            "seed": seed,
            "food": game.score,
            "steps": game.steps,
            "alive_at_horizon": not game.terminal,
            "outcome": game.outcome or "horizon_survived",
        }
        for seed, game in zip(seeds, games)
    ]
    return {
        "policy_id": policy.policy_id,
        "games": len(records),
        "mean_food": float(np.mean([row["food"] for row in records])),
        "median_food": float(np.median([row["food"] for row in records])),
        "mean_steps": float(np.mean([row["steps"] for row in records])),
        "alive_at_horizon": float(
            np.mean([row["alive_at_horizon"] for row in records])
        ),
        "collision_rate": float(
            np.mean(["collision" in row["outcome"] for row in records])
        ),
        "records": records,
    }


def promotion_decision(incumbent, candidate, minimum_gain, z_value):
    if [row["seed"] for row in incumbent["records"]] != [
        row["seed"] for row in candidate["records"]
    ]:
        raise ValueError("Promotion games must be paired by seed")
    deltas = np.asarray(
        [
            new["food"] - old["food"]
            for old, new in zip(incumbent["records"], candidate["records"])
        ],
        dtype=float,
    )
    mean = float(deltas.mean())
    standard_error = float(deltas.std(ddof=1) / math.sqrt(len(deltas))) if len(deltas) > 1 else 0.0
    lower_bound = mean - z_value * standard_error
    return {
        "accepted": bool(lower_bound > minimum_gain),
        "paired_mean_food_gain": mean,
        "paired_standard_error": standard_error,
        "one_sided_lower_bound": lower_bound,
        "z_value": z_value,
        "minimum_gain": minimum_gain,
    }
