"""Online rollouts of the 24 game under the frozen policy of the current generation.

Each step scores every action of every active puzzle in batched forwards, so one pass
produces both the training signal and the distribution the action is drawn from. The
label of a row is the observed event: did that episode finish on the target?
"""

import math
import random

import numpy as np
import torch

from .dataset import collate
from .env import State
from .evaluation import inference_precision
from .puzzles import is_solvable
from .serialization import question_row
from .utils import stream_seed


def score_candidates(model, tokenizer, entries, config):
    """Per-action success probability for ``(state, actions)`` entries.

    Returns one ``{action_key: p_yes}`` mapping per entry, in batched forwards.
    """
    rows, owners = [], []
    for index, (state, actions) in enumerate(entries):
        for action in actions:
            rows.append(question_row(state, action, config["policy_id"]))
            owners.append((index, action["key"]))
    if not rows:
        return [dict() for _ in entries]
    device = next(model.parameters()).device
    model.eval()
    probabilities = []
    micro = min(config["inference_questions"], len(rows))
    with torch.no_grad(), inference_precision(config["inference_precision"]):
        for offset in range(0, len(rows), micro):
            batch = collate(
                rows[offset : offset + micro], tokenizer, config["max_length"]
            )
            with torch.autocast(
                device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda" and config["inference_precision"] == "bf16",
            ):
                logp = model(batch)
            probabilities.extend(logp.exp().cpu().tolist())
    result = [dict() for _ in entries]
    for (index, key), distribution in zip(owners, probabilities):
        result[index][key] = distribution[0]
    return result


def action_distribution(probabilities, actions, temperature):
    """Normalize per-action success probabilities into a policy over actions."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    logits = np.array(
        [math.log(max(probabilities[action["key"]], 1e-30)) for action in actions]
    )
    logits = logits / temperature
    logits -= logits.max()
    weights = np.exp(logits)
    weights /= weights.sum()
    return {action["key"]: float(weight) for action, weight in zip(actions, weights)}


def choose(distribution, actions, rng, mode):
    if mode == "greedy":
        best = max(distribution.values())
        tied = [action for action in actions if distribution[action["key"]] == best]
        return tied[0] if len(tied) == 1 else rng.choice(tied)
    if mode != "sample":
        raise ValueError(f"Unknown rollout mode: {mode}")
    keys = [action["key"] for action in actions]
    drawn = rng.choices(keys, weights=[distribution[key] for key in keys])[0]
    return next(action for action in actions if action["key"] == drawn)


def pick_action(state, actions, probabilities, config, rng, mode):
    """Model policy, optionally mixed with the solver for the initial policy pi0."""
    distribution = action_distribution(probabilities, actions, config["temperature"])
    if config.get("solver_mix", 0.0) > 0.0 and rng.random() < config["solver_mix"]:
        solving = [
            action
            for action in actions
            if is_solvable(action["remaining"], state.target)
        ]
        if solving:
            return rng.choice(solving), distribution, "solver"
    return choose(distribution, actions, rng, mode), distribution, "model"


def rollout_batch(model, tokenizer, puzzles, config, seeds, mode="sample"):
    """One episode per ``(puzzle, seed)``; all episodes advance in lockstep."""
    if len(puzzles) != len(seeds):
        raise ValueError("Every rollout needs one seed")
    states = [State.initial(puzzle) for puzzle in puzzles]
    rngs = [random.Random(seed) for seed in seeds]
    records = [
        dict(puzzle=puzzle, steps=[], solved=False, answer=None) for puzzle in puzzles
    ]
    active = list(range(len(states)))
    while active:
        entries, choices = [], []
        for index in active:
            state = states[index]
            if state.finished:
                records[index]["answer"] = str(state.current)
                records[index]["solved"] = bool(state.solved)
                continue
            entries.append((state, state.actions()))
            choices.append(index)
        active = [index for index in active if not states[index].finished]
        if not entries:
            continue
        probabilities = score_candidates(model, tokenizer, entries, config)
        for slot, index in enumerate(choices):
            state, actions = entries[slot]
            action, distribution, controller = pick_action(
                state, actions, probabilities[slot], config, rngs[index], mode
            )
            records[index]["steps"].append(
                dict(
                    state=state,
                    actions=actions,
                    distribution=distribution,
                    chosen=action["key"],
                    action=action,
                    controller=controller,
                    entropy=float(
                        -sum(p * math.log(max(p, 1e-30)) for p in distribution.values())
                    ),
                    p_yes=probabilities[slot][action["key"]],
                )
            )
            states[index] = state.apply(action)
    for index, state in enumerate(states):
        if records[index]["answer"] is None:
            records[index]["answer"] = str(state.current)
            records[index]["solved"] = bool(state.solved)
    return records


def record_json(record):
    """JSON-safe trajectory summary."""
    puzzle = record["puzzle"]
    return dict(
        puzzle_id=puzzle["puzzle_id"],
        numbers=list(puzzle["numbers"]),
        target=puzzle["target"],
        answer=record["answer"],
        solved=record["solved"],
        controllers=[step["controller"] for step in record["steps"]],
        steps=[
            dict(
                state_id=step["state"].state_id(),
                depth=step["state"].depth,
                remaining=[str(value) for value in step["state"].values],
                step_history=[f"{s['left']} {s['op']} {s['right']} = {s['result']}" for s in step["state"].history],
                chosen=step["chosen"],
                action_text=f"{step['action']['left']} {step['action']['op']} "
                f"{step['action']['right']} = {step['action']['value']}",
                controller=step["controller"],
                entropy=step["entropy"],
                p_yes=step["p_yes"],
                distribution=step["distribution"],
            )
            for step in record["steps"]
        ],
    )


def training_rows(records, policy_id):
    """Deduplicated training rows: one observed outcome per ``(state, action)``."""
    rows, seen, duplicates = [], set(), 0
    for record in records:
        outcome = "yes" if record["solved"] else "no"
        for step in record["steps"]:
            key = (step["state"].state_id(), step["chosen"])
            if key in seen:
                duplicates += 1
                continue
            seen.add(key)
            row = question_row(
                step["state"], step["action"], policy_id, candidate_id="yes"
            )
            rows.append(
                row
                | dict(
                    observed_outcome=outcome,
                    controller=step["controller"],
                )
            )
    return rows, duplicates


def rollout_seeds(puzzles, config, generation, repeats):
    """Deterministic seeds for every rollout of this generation."""
    ordered, seeds = [], []
    for repeat in range(repeats):
        for puzzle in puzzles:
            ordered.append(puzzle)
            seeds.append(
                stream_seed(
                    config["seed"], "rollout", generation, puzzle["puzzle_id"], repeat
                )
            )
    return ordered, seeds


def collect(model, tokenizer, puzzles, config, generation, mode="sample"):
    """Roll out every puzzle ``rollouts_per_puzzle`` times and build training rows."""
    import time

    repeats = config["rollouts_per_puzzle"]
    ordered, seeds = rollout_seeds(puzzles, config, generation, repeats)
    started = time.monotonic()
    records = rollout_batch(model, tokenizer, ordered, config, seeds, mode=mode)
    rows, duplicates = training_rows(records, config["policy_id"])
    metrics = dict(
        generation=generation,
        puzzles=len(puzzles),
        rollouts=len(records),
        states_visited=sum(len(record["steps"]) for record in records),
        training_rows=len(rows),
        duplicate_rows=duplicates,
        solved_rate=_rate(sum(record["solved"] for record in records), len(records)),
        row_yes_rate=_rate(
            sum(row["observed_outcome"] == "yes" for row in rows), len(rows)
        ),
        mean_p_yes=float(
            np.mean([step["p_yes"] for record in records for step in record["steps"]])
        ),
        mean_action_entropy=float(
            np.mean([step["entropy"] for record in records for step in record["steps"]])
        ),
        solver_step_share=_rate(
            sum(step["controller"] == "solver" for record in records for step in record["steps"]),
            sum(len(record["steps"]) for record in records),
        ),
        collect_seconds=time.monotonic() - started,
    )
    return records, rows, metrics


def _rate(numerator, denominator):
    return float(numerator) / denominator if denominator else None
