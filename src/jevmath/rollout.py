"""Online rollouts under the frozen policy of the current generation.

Every step scores all structured candidates of a state in one batched forward. The
policy over actions is the normalized per-action success probability, so the same
forward pass yields both the training signal and the action distribution.
"""

import math
import random

import numpy as np
import torch

from .dataset import collate
from .env import State, forced_stop_value
from .evaluation import inference_precision
from .serialization import question_row


def score_candidates(model, tokenizer, entries, config):
    """Per-action success probability for ``(state, candidates)`` entries.

    Returns one ``{action_key: p_yes}`` mapping per entry, evaluated in batched
    forwards of complete candidate sets.
    """
    rows, owners = [], []
    for index, (state, candidates) in enumerate(entries):
        for candidate in candidates:
            rows.append(question_row(state, candidate, config["policy_id"]))
            owners.append((index, candidate["key"]))
    if not rows:
        return [dict() for _ in entries]
    device = next(model.parameters()).device
    model.eval()
    probabilities = []
    micro = min(config["inference_questions"], len(rows))
    with torch.no_grad(), inference_precision(config["inference_precision"]):
        for offset in range(0, len(rows), micro):
            batch = collate(rows[offset : offset + micro], tokenizer, config["max_length"])
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


def action_distribution(probabilities, candidates, temperature):
    """Normalize per-action success probabilities into a policy over actions."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    logits = np.array(
        [math.log(max(probabilities[candidate["key"]], 1e-30)) for candidate in candidates]
    )
    logits = logits / temperature
    weights = np.exp(logits - logits.max())
    weights = weights / weights.sum()
    return {
        candidate["key"]: float(weight)
        for candidate, weight in zip(candidates, weights)
    }


def choose(distribution, candidates, rng, mode):
    if mode not in ("sample", "greedy"):
        raise ValueError(mode)
    if mode == "greedy":
        key = max(distribution, key=distribution.__getitem__)
    else:
        keys = [candidate["key"] for candidate in candidates]
        key = rng.choices(keys, weights=[distribution[k] for k in keys])[0]
    return next(candidate for candidate in candidates if candidate["key"] == key)


def rollout_batch(model, tokenizer, problems, config, seeds, mode="sample"):
    """One trajectory per ``(problem, seed)``; all rollouts advance in lockstep.

    Returns a list of records with the chosen actions, the distributions they were drawn
    from, the reported final answer and whether it matches the gold answer.
    """
    if len(problems) != len(seeds):
        raise ValueError("Every rollout needs one seed")
    states = [State.initial(row) for row in problems]
    rngs = [random.Random(seed) for seed in seeds]
    records = [dict(steps=[], forced=False, answer=None) for _ in states]
    active = list(range(len(states)))
    step_index = 0
    while active:
        entries, choices = [], []
        for index in active:
            state = states[index]
            candidates = state.candidates()
            if len(state.quantities) == 1 or state.depth >= config["max_steps"]:
                records[index]["answer"] = forced_stop_value(state)
                records[index]["forced"] = True
                continue
            entries.append((state, candidates))
            choices.append(index)
        if not entries:
            active = [index for index in active if records[index]["answer"] is None]
            continue
        probabilities = score_candidates(model, tokenizer, entries, config)
        advanced = []
        for slot, index in enumerate(choices):
            state, candidates = entries[slot]
            distribution = action_distribution(
                probabilities[slot], candidates, config["temperature"]
            )
            candidate = choose(distribution, candidates, rngs[index], mode)
            records[index]["steps"].append(
                dict(
                    state=state,
                    candidates=candidates,
                    distribution=distribution,
                    chosen=candidate["key"],
                    kind=candidate["kind"],
                    entropy=float(
                        -sum(p * math.log(max(p, 1e-30)) for p in distribution.values())
                    ),
                )
            )
            if candidate["kind"] == "STOP":
                records[index]["answer"] = candidate["value"]
            else:
                states[index] = state.combine(candidate)
                advanced.append(index)
        active = advanced
        step_index += 1
        if step_index > config["max_steps"] + 1:
            raise RuntimeError("Rollout exceeded its step budget")
    for index, record in enumerate(records):
        record["problem_id"] = problems[index]["id"]
        record["correct"] = record["answer"] == problems[index]["gold"]
        record["answer"] = str(record["answer"])
        record["gold"] = str(problems[index]["gold"])
    return records


def record_json(record):
    """JSON-safe trajectory summary; the in-memory record keeps its State objects."""
    return dict(
        problem_id=record["problem_id"],
        gold=record["gold"],
        answer=record["answer"],
        correct=record["correct"],
        forced=record["forced"],
        steps=[
            dict(
                depth=step["state"].depth,
                state_id=step["state"].state_id(),
                pool=[(str(value), label) for value, label in step["state"].quantities],
                step_history=list(step["state"].steps),
                chosen=step["chosen"],
                kind=step["kind"],
                entropy=step["entropy"],
                distribution=dict(step["distribution"]),
                candidates=[candidate["key"] for candidate in step["candidates"]],
            )
            for step in record["steps"]
        ],
    )


def trajectory_rows(record, problems, policy_id):
    """Observed-event rows: the taken action of each step, labelled by the final outcome."""
    outcome = "yes" if record["correct"] else "no"
    row = next(item for item in problems if item["id"] == record["problem_id"])
    rows = []
    for step in record["steps"]:
        state = step["state"]
        candidate = next(
            item for item in state.candidates() if item["key"] == step["chosen"]
        )
        rows.append(
            question_row(state, candidate, policy_id)
            | dict(observed_outcome=outcome, problem_id=record["problem_id"])
        )
    return rows


def collect(model, tokenizer, problems, config, seeds, mode="sample"):
    """Roll out ``config.rollouts_per_problem`` trajectories for every sampled problem."""
    grouped = {}
    for index, problem in enumerate(problems):
        grouped.setdefault(problem["id"], []).append(index)
    rollout_problems, rollout_seeds = [], []
    for problem_id, indices in grouped.items():
        for repeat in range(config["rollouts_per_problem"]):
            rollout_problems.append(problems[indices[0]])
            rollout_seeds.append(seeds[(problem_id, repeat)])
    records = rollout_batch(
        model, tokenizer, rollout_problems, config, rollout_seeds, mode
    )
    training = []
    for record in records:
        training.extend(trajectory_rows(record, problems, config["policy_id"]))
    return records, training
