from fractions import Fraction

import pytest

from jevmath.env import CandidateError, State, apply_op, forced_stop_value
from jevmath.gsm8k import final_answer, question_numbers

ROW = dict(
    id="demo",
    question="Ann has 3 apples and buys 4 more, then gives away 2.",
    numbers=((Fraction(3), "q1"), (Fraction(4), "q2"), (Fraction(2), "q3")),
    gold=Fraction(5),
)


def test_calculator_operations():
    assert apply_op("ADD", Fraction(3), Fraction(4)) == 7
    assert apply_op("SUB", Fraction(3), Fraction(4)) == -1
    assert apply_op("MUL", Fraction(3), Fraction(4)) == 12
    assert apply_op("DIV", Fraction(6), Fraction(4)) == Fraction(3, 2)
    with pytest.raises(ZeroDivisionError):
        apply_op("DIV", Fraction(1), Fraction(0))
    with pytest.raises(CandidateError):
        apply_op("POW", Fraction(1), Fraction(1))


def test_candidates_are_well_formed():
    state = State.initial(ROW)
    candidates = state.candidates()
    keys = [candidate["key"] for candidate in candidates]
    assert len(keys) == len(set(keys))
    combines = [candidate for candidate in candidates if candidate["kind"] == "COMBINE"]
    stops = [candidate for candidate in candidates if candidate["kind"] == "STOP"]
    assert len(stops) == len(state.quantities)
    for candidate in combines:
        assert candidate["i"] != candidate["j"]
        if candidate["op"] in ("ADD", "MUL"):
            assert candidate["i"] < candidate["j"]
        assert candidate["value"] == apply_op(
            candidate["op"],
            state.quantities[candidate["i"]][0],
            state.quantities[candidate["j"]][0],
        )
        assert state.describe(candidate).endswith(str(candidate["value"]))


def test_combine_shrinks_the_pool_and_records_steps():
    state = State.initial(ROW)
    combine = next(
        candidate
        for candidate in state.candidates()
        if candidate["kind"] == "COMBINE" and candidate["op"] == "ADD"
    )
    advanced = state.combine(combine)
    assert len(advanced.quantities) == len(state.quantities) - 1
    assert advanced.quantities[-1][0] == combine["value"]
    assert advanced.depth == 1 and len(advanced.steps) == 1
    assert advanced.steps[0]["result"] == str(combine["value"])
    with pytest.raises(CandidateError):
        state.combine(next(c for c in state.candidates() if c["kind"] == "STOP"))


def test_scripted_solution_reaches_the_gold_answer():
    """The environment can produce a correct episode, so Y=1 is reachable."""
    state = State.initial(ROW)
    add = next(
        candidate
        for candidate in state.candidates()
        if candidate["key"] == "COMBINE:ADD:0:1"
    )
    state = state.combine(add)
    subtract = next(
        candidate
        for candidate in state.candidates()
        if candidate["kind"] == "COMBINE"
        and candidate["op"] == "SUB"
        and candidate["value"] == ROW["gold"]
    )
    state = state.combine(subtract)
    assert len(state.quantities) == 1
    assert forced_stop_value(state) == ROW["gold"]
    assert state.is_correct(forced_stop_value(state))


def test_pool_collapse_forces_a_deterministic_answer():
    state = State.initial(ROW)
    while len(state.quantities) > 1:
        state = state.combine(state.candidates()[0])
    assert forced_stop_value(state) == state.quantities[0][0]
    assert len(state.candidates()) == 1 and state.candidates()[0]["kind"] == "STOP"


def test_gsm8k_parsing():
    assert final_answer("steps...\n#### 1,234") == Fraction(1234)
    assert final_answer("#### 2.5") == Fraction(5, 2)
    with pytest.raises(ValueError):
        final_answer("no marker here")
    assert question_numbers("Tom has 3 cats and 12.5 dollars, twice 4") == (
        (Fraction(3), "q1"),
        (Fraction(25, 2), "q2"),
        (Fraction(4), "q3"),
    )
