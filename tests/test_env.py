"""24 game states: exact rationals, action dedupe, three operations, no stop action."""

from fractions import Fraction

import pytest

from jevmath.env import ActionError, State, action_key, solved_after
from jevmath.puzzles import is_solvable


def puzzle(numbers, target=24):
    return dict(puzzle_id="_".join(map(str, numbers)), numbers=list(numbers), target=target)


def test_initial_state_is_a_sorted_multiset_of_fractions():
    state = State.initial(puzzle([8, 3, 8, 3]))
    assert state.values == (Fraction(3), Fraction(3), Fraction(8), Fraction(8))
    assert state.remaining_text == "3 3 8 8"
    assert state.depth == 0 and state.history == ()
    assert state.finished is False
    with pytest.raises(ActionError):
        state.current


def test_every_action_uses_two_values_and_produces_an_exact_result():
    state = State.initial(puzzle([1, 5, 5, 5]))
    for action in state.actions():
        assert action["remaining"] == state.remaining_after(
            action["left"], action["right"], action["value"]
        )
        assert action["value"] == {
            "+": action["left"] + action["right"],
            "-": action["left"] - action["right"],
            "*": action["left"] * action["right"],
            "/": action["left"] / action["right"],
        }[action["op"]]
        assert len(action["remaining"]) == 3


def test_fractional_intermediate_values_stay_exact():
    state = State.initial(puzzle([3, 3, 8, 8]))
    action = next(a for a in state.actions() if a["key"] == "/:8:3")
    assert action["value"] == Fraction(8, 3)
    after = state.apply(action)
    assert after.values == (Fraction(8, 3), Fraction(3), Fraction(8))
    assert after.state_id().endswith("8/3|3|8")
    assert solved_after(state, action) is True


def test_commutative_and_duplicate_actions_collapse():
    state = State.initial(puzzle([6, 6, 6, 6]))
    keys = [action["key"] for action in state.actions()]
    assert len(keys) == len(set(keys))
    assert keys.count("+:6:6") == 1
    assert keys.count("*:6:6") == 1
    assert keys.count("-:6:6") == 1
    assert keys.count("/:6:6") == 1
    assert len(keys) == 4
    bigger = State.initial(puzzle([1, 2, 3, 4]))
    assert len(bigger.actions()) == 3 * 4 * 3  # 3 * m * (m - 1) with m = 4


def test_division_by_zero_is_never_offered():
    state = State.initial(puzzle([1, 1, 2, 3]))
    zero = state.apply(next(a for a in state.actions() if a["key"] == "-:1:1"))
    assert Fraction(0) in zero.values
    assert all(a["right"] != 0 for a in zero.actions() if a["op"] == "/")
    with pytest.raises(ZeroDivisionError):
        from jevmath.puzzles import apply_op

        apply_op("/", Fraction(1), Fraction(0))


def test_episode_is_three_operations_and_there_is_no_stop_action():
    state = State.initial(puzzle([1, 1, 1, 8]))
    depth = 0
    while not state.finished:
        assert all(action["op"] in ("+", "-", "*", "/") for action in state.actions())
        action = next(a for a in state.actions() if a["key"] == "+:1:1") if depth == 0 else state.actions()[0]
        state = state.apply(action)
        depth += 1
    assert depth == 3
    assert state.depth == 3
    assert len(state.history) == 3
    assert state.finished and state.current == Fraction(1) + Fraction(1) + Fraction(1) + Fraction(8)


def test_a_wrong_final_value_is_not_solved():
    state = State.initial(puzzle([3, 3, 8, 8]))
    for key in ("+:3:3", "+:6:8", "+:8:14"):
        state = state.apply(next(a for a in state.actions() if a["key"] == key))
    assert state.finished and state.current == Fraction(22)
    assert state.solved is False


def test_a_full_solving_episode_reaches_the_target():
    state = State.initial(puzzle([3, 3, 8, 8]))
    for key in ("/:8:3", "-:3:8/3", "/:8:1/3"):
        state = state.apply(next(a for a in state.actions() if a["key"] == key))
    assert state.finished and state.current == Fraction(24)
    assert state.solved is True
    assert [step["text"] for step in state.history] == ["8 / 3 = 8/3", "3 - 8/3 = 1/3", "8 / 1/3 = 24"]


def test_unsolvable_actions_are_reported_by_the_solver():
    state = State.initial(puzzle([3, 3, 8, 8]))
    action = next(a for a in state.actions() if a["key"] == "*:3:8")
    assert action["value"] == Fraction(24)
    assert solved_after(state, action) is False  # the remaining 3 and 8 cannot reach 24


def test_action_key_orders_commutative_operands():
    assert action_key("+", Fraction(8), Fraction(3)) == "+:3:8"
    assert action_key("*", Fraction(3), Fraction(8)) == "*:3:8"
    assert action_key("-", Fraction(8), Fraction(3)) == "-:8:3"
    assert action_key("/", Fraction(3), Fraction(8)) == "/:3:8"
