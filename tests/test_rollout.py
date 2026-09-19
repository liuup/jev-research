"""Rollouts: determinism, lockstep episodes, the solver-mixed pi0 and row labelling."""

from pathlib import Path

import pytest

from jevmath.env import State
from jevmath.puzzles import load_puzzles
from jevmath.rollout import (
    action_distribution,
    choose,
    collect,
    pick_action,
    rollout_batch,
    rollout_seeds,
    training_rows,
)
from jevmath.tiny import tiny_model_and_tokenizer

DATA = Path("data/24game")


def config(**overrides):
    base = dict(
        seed=17,
        policy_id="24game:seed17:g0",
        solver_mix=0.0,
        temperature=1.0,
        inference_questions=16,
        inference_precision="fp32",
        max_length=512,
        rollouts_per_puzzle=2,
    )
    return base | overrides


def puzzles(count=4):
    if not (DATA / "train.jsonl").exists():
        pytest.skip("dataset not built")
    return load_puzzles(DATA / "train.jsonl")[:count]


def test_action_distribution_normalizes_success_probabilities():
    actions = State.initial(puzzles(1)[0]).actions()
    probabilities = {action["key"]: 0.25 for action in actions}
    distribution = action_distribution(probabilities, actions, 1.0)
    assert abs(sum(distribution.values()) - 1.0) < 1e-9
    assert all(abs(value - 1 / len(actions)) < 1e-9 for value in distribution.values())
    peaked = action_distribution(
        {action["key"]: (0.9 if index == 0 else 0.01) for index, action in enumerate(actions)},
        actions,
        1.0,
    )
    assert max(peaked.values()) == peaked[actions[0]["key"]]


def test_choose_is_reproducible_and_greedy_picks_the_best():
    import random

    actions = State.initial(puzzles(1)[0]).actions()
    distribution = {action["key"]: 1 / len(actions) for action in actions}
    first = choose(distribution, actions, random.Random(3), "sample")
    second = choose(distribution, actions, random.Random(3), "sample")
    assert first["key"] == second["key"]
    greedy = {action["key"]: 0.0 for action in actions}
    greedy[actions[2]["key"]] = 1.0
    assert choose(greedy, actions, random.Random(0), "greedy")["key"] == actions[2]["key"]
    with pytest.raises(ValueError):
        choose(distribution, actions, random.Random(0), "noisy")


def test_solver_mix_always_solves_solvable_puzzles():
    model, tokenizer = tiny_model_and_tokenizer()
    batch = puzzles(6)
    ordered, seeds = rollout_seeds(batch, config(solver_mix=1.0), 0, 2)
    records = rollout_batch(model, tokenizer, ordered, config(solver_mix=1.0), seeds)
    assert all(all(step["controller"] == "solver" for step in record["steps"]) for record in records)
    assert all(record["solved"] for record in records)
    assert all(len(record["steps"]) == 3 for record in records)
    assert all(record["answer"] == "24" for record in records)


def test_rollouts_are_deterministic_for_the_same_seeds():
    model, tokenizer = tiny_model_and_tokenizer()
    batch = puzzles(4)
    ordered, seeds = rollout_seeds(batch, config(solver_mix=0.5), 0, 2)
    first = rollout_batch(model, tokenizer, ordered, config(solver_mix=0.5), seeds)
    second = rollout_batch(model, tokenizer, ordered, config(solver_mix=0.5), seeds)
    assert [record["answer"] for record in first] == [record["answer"] for record in second]
    assert [len(record["steps"]) for record in first] == [len(record["steps"]) for record in second]
    assert [step["chosen"] for record in first for step in record["steps"]] == [
        step["chosen"] for record in second for step in record["steps"]
    ]


def test_greedy_rollouts_are_deterministic_and_sample_mode_explores():
    model, tokenizer = tiny_model_and_tokenizer()
    batch = puzzles(4)
    ordered, seeds = rollout_seeds(batch, config(), 0, 1)
    greedy_a = rollout_batch(model, tokenizer, ordered, config(), seeds, mode="greedy")
    greedy_b = rollout_batch(model, tokenizer, ordered, config(), seeds, mode="greedy")
    assert [step["chosen"] for r in greedy_a for step in r["steps"]] == [
        step["chosen"] for r in greedy_b for step in r["steps"]
    ]
    sampled = [
        step["chosen"]
        for repeat in range(6)
        for record in rollout_batch(
            model,
            tokenizer,
            ordered,
            config(),
            [seed + repeat * 10**6 for seed in seeds],
        )
        for step in record["steps"]
    ]
    assert len(set(sampled)) > len({step["chosen"] for r in greedy_a for step in r["steps"]})


def test_training_rows_keep_one_outcome_per_state_and_action():
    state = State.initial(puzzles(1)[0])
    action = state.actions()[0]
    step = dict(state=state, action=action, chosen=action["key"], controller="model")
    records = [
        dict(puzzle=state.puzzle, solved=True, steps=[step]),
        dict(puzzle=state.puzzle, solved=False, steps=[step]),
        dict(puzzle=state.puzzle, solved=True, steps=[step]),
    ]
    rows, duplicates = training_rows(records, "24game:seed17:g0")
    assert len(rows) == 1 and duplicates == 2
    assert rows[0]["observed_outcome"] == "yes"
    assert rows[0]["candidate_ids"] == ["yes", "no"]
    assert rows[0]["state_id"] == state.state_id()


def test_collect_reports_labels_consistent_with_the_trajectories():
    model, tokenizer = tiny_model_and_tokenizer()
    records, rows, metrics = collect(
        model, tokenizer, puzzles(8), config(solver_mix=0.5), 0, mode="sample"
    )
    assert metrics["rollouts"] == 16
    assert metrics["training_rows"] == len(rows)
    assert metrics["duplicate_rows"] >= 0
    assert 0.0 <= metrics["solved_rate"] <= 1.0
    assert 0.0 <= metrics["solver_step_share"] <= 1.0
    for row in rows:
        assert row["observed_outcome"] == (
            "yes" if _trajectory_solved(records, row) else "no"
        )
    assert metrics["row_yes_rate"] == pytest.approx(
        sum(row["observed_outcome"] == "yes" for row in rows) / len(rows)
    )


def _trajectory_solved(records, row):
    for record in records:
        for step in record["steps"]:
            if step["state"].state_id() == row["state_id"] and step["chosen"] == row["action_key"]:
                return record["solved"]
    raise AssertionError("row is not attached to any trajectory")


def test_pick_action_falls_back_to_the_model_when_nothing_is_solvable():
    import random

    from jevmath.puzzles import load_puzzles as _load

    unsolvable = _load(DATA / "unsolvable.jsonl")[0]
    state = State.initial(unsolvable)
    actions = state.actions()
    probabilities = {action["key"]: 0.5 for action in actions}
    action, distribution, controller = pick_action(
        state, actions, probabilities, config(solver_mix=1.0), random.Random(1), "greedy"
    )
    assert controller == "model"
    assert len(distribution) == len(actions)
