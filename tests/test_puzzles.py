"""Puzzle generation, the exact solver, the oracle table and the splits."""

from collections import Counter
from fractions import Fraction
from pathlib import Path

import pytest

from jevmath.env import State
from jevmath.puzzles import (
    SOLVABLE,
    SPLIT_SIZES,
    UNSOLVABLE,
    build_puzzles,
    canonical_key,
    evaluate_expression,
    expression_solutions,
    first_step_actions,
    hands,
    is_solvable,
    load_puzzles,
    puzzle_id,
    render,
    solution_strings,
    split_puzzles,
    verify_manifest,
    write_dataset,
)

DATA = Path("data/24game")


def expression_leaves(expression):
    if expression[0] == "num":
        return [expression[1]]
    return expression_leaves(expression[2]) + expression_leaves(expression[3])


def test_hand_space_matches_the_plan():
    combos = hands()
    assert len(combos) == 1_820
    assert len(set(combos)) == 1_820
    assert all(list(hand) == sorted(hand) for hand in combos)
    assert all(1 <= number <= 13 for hand in combos for number in hand)
    assert Counter(len(hand) for hand in combos) == {4: 1_820}


def test_solvable_counts_match_the_plan():
    puzzles = build_puzzles()
    solvable = [puzzle for puzzle in puzzles if puzzle["solvable"]]
    assert len(puzzles) == 1_820
    assert len(solvable) == SOLVABLE == 1_362
    assert len(puzzles) - len(solvable) == UNSOLVABLE == 458


def test_every_solution_evaluates_to_the_target():
    checked = 0
    for hand in hands():
        expected = Counter(Fraction(value) for value in hand)
        for expression in expression_solutions(hand):
            solution = render(expression)
            assert evaluate_expression(solution) == 24, (hand, solution)
            assert Counter(expression_leaves(expression)) == expected, (hand, solution)
            checked += 1
    assert checked > 1_000


def test_known_hand_has_the_expected_unique_solution():
    puzzle = next(p for p in build_puzzles() if p["puzzle_id"] == "3_3_8_8")
    assert puzzle["solvable"] is True
    assert puzzle["solution_count"] == 1
    assert puzzle["example_solution"] == "8 / (3 - 8 / 3)"
    assert solution_strings([3, 3, 8, 8]) == {puzzle["example_solution"]}


def test_unsolvable_hand_has_no_solutions():
    assert is_solvable([1, 1, 1, 1]) is False
    assert solution_strings([1, 1, 1, 1]) == set()
    assert is_solvable([3, 3, 8, 8]) is True
    assert is_solvable([8, 8, 8, 8]) is False  # every number must be used exactly once
    assert is_solvable([1, 1, 1, 8]) is True  # 8 * (1 + 1 + 1)


def test_first_step_oracle_agrees_with_the_solver():
    actions = first_step_actions([3, 3, 8, 8])
    by_key = {action["action_key"]: action for action in actions}
    assert len(actions) == len(by_key) == 14  # 2 + 2 + 4 + 4 pairs, plus 2 for equal values
    solving = [action for action in actions if action["solvable_after"]]
    assert [action["action_key"] for action in solving] == ["/:8:3"]
    assert solving[0]["remaining"] == ["8/3", "3", "8"]
    assert by_key["*:3:8"]["solvable_after"] is False  # 3 * 8 leaves 3 8 -> dead end
    for action in actions:
        assert Fraction(action["result"]) == evaluate_expression(
            f"{action['left']} {action['op']} {action['right']}"
        )


def test_actions_are_deduplicated_and_skip_division_by_zero():
    actions = first_step_actions([1, 1, 2, 3])
    keys = [action["action_key"] for action in actions]
    assert len(keys) == len(set(keys))
    assert keys.count("+:1:1") == 1
    assert keys.count("-:1:1") == 1
    assert keys.count("/:1:1") == 1
    assert keys.count("*:1:1") == 1
    state = State.initial(dict(puzzle_id="x", numbers=[1, 1, 2, 3], target=24))
    zero = state.apply(next(a for a in state.actions() if a["key"] == "-:1:1"))
    assert zero.values == (0, 2, 3)
    assert all(a["right"] != 0 for a in zero.actions() if a["op"] == "/")
    keys = sorted(a["key"] for a in zero.actions())
    assert len(keys) == 16 and len(set(keys)) == 16
    assert "/:2:0" not in keys and "/:3:0" not in keys  # a zero divisor is never offered
    assert "/:0:2" in keys and "/:0:3" in keys  # zero may itself be divided


def test_split_sizes_are_disjoint_and_cover_the_solvable_set():
    solvable = [puzzle for puzzle in build_puzzles() if puzzle["solvable"]]
    splits = split_puzzles(solvable, seed=17)
    assert {name: len(rows) for name, rows in splits.items()} == SPLIT_SIZES
    ids = {name: {puzzle["puzzle_id"] for puzzle in rows} for name, rows in splits.items()}
    assert not ids["train"] & ids["dev"]
    assert not ids["train"] & ids["test"]
    assert not ids["dev"] & ids["test"]
    assert len(ids["train"] | ids["dev"] | ids["test"]) == 1_362


def test_split_uses_the_canonical_hand_as_group_key():
    solvable = [puzzle for puzzle in build_puzzles() if puzzle["solvable"]]
    assert len({canonical_key(puzzle["numbers"]) for puzzle in solvable}) == len(solvable)
    assert canonical_key([8, 3, 3, 8]) == (3, 3, 8, 8)
    assert puzzle_id([8, 3, 8, 3]) == "8_3_8_3"
    with pytest.raises(ValueError):
        split_puzzles(solvable, sizes=dict(SPLIT_SIZES, train=1))


def test_generated_files_match_the_manifest(tmp_path):
    manifest = write_dataset(tmp_path, seed=17)
    assert manifest["counts"] == dict(
        hands=1_820, solvable=1_362, unsolvable=458, train=954, dev=204, test=204
    )
    assert manifest["rules"]["stop_action"] is False
    assert manifest["rules"]["operations_per_episode"] == 3
    for name in ("train.jsonl", "dev.jsonl", "test.jsonl", "unsolvable.jsonl", "oracle_actions.jsonl"):
        assert (tmp_path / name).exists(), name
    verify_manifest(tmp_path, manifest)
    (tmp_path / "train.jsonl").write_text("")
    with pytest.raises(ValueError, match="manifest"):
        verify_manifest(tmp_path, manifest)


@pytest.mark.skipif(not (DATA / "manifest.json").exists(), reason="dataset not built")
def test_repository_dataset_is_intact():
    manifest = verify_manifest(DATA)
    train = load_puzzles(DATA / "train.jsonl")
    dev = load_puzzles(DATA / "dev.jsonl")
    test = load_puzzles(DATA / "test.jsonl")
    unsolvable = load_puzzles(DATA / "unsolvable.jsonl")
    assert (len(train), len(dev), len(test), len(unsolvable)) == (954, 204, 204, 458)
    assert all(puzzle["solvable"] for puzzle in train + dev + test)
    assert all(not puzzle["solvable"] for puzzle in unsolvable)
    assert all(puzzle["example_solution"] for puzzle in train + dev + test)
    assert manifest["counts"]["solvable"] == 1_362
