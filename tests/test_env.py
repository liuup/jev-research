from collections import Counter

import pytest

from jev2048.env import ACTIONS, Game, merge, moved
from jev2048.heuristic import HeuristicPolicy


@pytest.mark.parametrize(
    "row,out,score",
    [
        ([2, 2, 2, 2], [4, 4, 0, 0], 8),
        ([4, 4, 8, 8], [8, 16, 0, 0], 24),
        ([2, 2, 4, 0], [4, 4, 0, 0], 4),
        ([4, 0, 4, 4], [8, 4, 0, 0], 8),
        ([0, 0, 0, 0], [0, 0, 0, 0], 0),
        ([2, 4, 8, 16], [2, 4, 8, 16], 0),
        ([2, 2, 4, 4], [4, 8, 0, 0], 12),
        ([2, 0, 0, 2], [4, 0, 0, 0], 4),
    ],
)
def test_merge(row, out, score):
    assert merge(tuple(row)) == (tuple(out), score)


@pytest.mark.parametrize("action", ACTIONS)
def test_directions(action):
    board = ((2, 2, 0, 0),) * 4
    b, s = moved(board, action)
    assert s == 16
    assert sum(map(sum, b)) == sum(map(sum, board))
    if action == "LEFT":
        assert b[0] == (4, 0, 0, 0)
    if action == "RIGHT":
        assert b[0] == (0, 0, 0, 4)
    if action == "UP":
        assert b[0] == (4, 4, 0, 0) and b[3] == (0, 0, 0, 0)
    if action == "DOWN":
        assert b[3] == (4, 4, 0, 0) and b[0] == (0, 0, 0, 0)


def test_seed_clone_invalid_terminal():
    a, b = Game(17), Game(17)
    p = HeuristicPolicy()
    for _ in range(30):
        assert a.state() == b.state()
        assert p.choose(a) == p.choose(b)
        c = a.clone()
        action = p.choose(a)
        a.step(action)
        b.step(action)
        c.step(action)
        assert a.state() == c.state()
    full = Game(board=((2, 4, 2, 4), (4, 2, 4, 2), (2, 4, 2, 4), (4, 2, 4, 2)))
    before = full.clone()
    assert full.terminal and full.legal_actions == []
    assert not full.step("LEFT")
    assert (
        full.state() == before.state() and full.rng.getstate() == before.rng.getstate()
    )


def test_spawn():
    g = Game(board=((2, 0, 0, 0),) * 4)
    assert g.step("RIGHT") and g.steps == 1
    assert sum(x > 0 for r in g.board for x in r) == 5


def test_legal_actions_and_score():
    g = Game(board=((2, 2, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0)), score=12)
    assert g.legal_actions == ["LEFT", "RIGHT", "DOWN"]
    assert g.step("LEFT") and g.score == 16 and g.max_tile == 4
    clone = g.clone()
    clone.step(clone.legal_actions[0])
    assert clone.steps == g.steps + 1


def test_spawn_distribution():
    counts = Counter()
    cells = Counter()
    for seed in range(5000):
        g = Game(seed, board=((0,) * 4,) * 4)
        g.spawn()
        counts[g.max_tile] += 1
        cells.update((i, j) for i in range(4) for j in range(4) if g.board[i][j])
    assert 0.08 < counts[4] / 5000 < 0.12
    assert all(240 < n < 390 for n in cells.values())
