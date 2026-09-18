"""Standard 2048; board entries are tile values, never exponents."""
import copy
import random
from functools import lru_cache

ACTIONS = ("LEFT", "RIGHT", "UP", "DOWN")

@lru_cache(maxsize=65536)
def merge(row):
    values = [x for x in row if x]
    out, score, i = [], 0, 0
    while i < len(values):
        if i + 1 < len(values) and values[i] == values[i + 1]:
            out.append(2 * values[i]); score += out[-1]; i += 2
        else:
            out.append(values[i]); i += 1
    return tuple(out + [0] * (4 - len(out))), score

def moved(board, action):
    if action not in ACTIONS:
        raise ValueError(action)
    rows = board if action in ACTIONS[:2] else tuple(zip(*board))
    reverse = action in ("RIGHT", "DOWN")
    result, score = [], 0
    for row in rows:
        new, gain = merge(tuple(reversed(row)) if reverse else tuple(row))
        result.append(tuple(reversed(new)) if reverse else new); score += gain
    return (tuple(result) if action in ACTIONS[:2] else tuple(zip(*result))), score

class Game:
    def __init__(self, seed=0, board=None, score=0, steps=0):
        self.rng = random.Random(seed)
        self.board = tuple(tuple(r) for r in board) if board is not None else ((0,)*4,)*4
        if len(self.board) != 4 or any(len(r) != 4 for r in self.board):
            raise ValueError("Expected 4x4 board")
        if any(x < 0 or (x and (x < 2 or x & (x-1))) for r in self.board for x in r):
            raise ValueError("Tiles must be zero or powers of two >= 2")
        self.score, self.steps = score, steps
        if board is None:
            self.spawn(); self.spawn()

    def clone(self):
        return copy.deepcopy(self)

    def spawn(self):
        empty = [(i,j) for i in range(4) for j in range(4) if not self.board[i][j]]
        if empty:
            i,j = self.rng.choice(empty)
            rows = [list(r) for r in self.board]
            rows[i][j] = 2 if self.rng.random() < 0.9 else 4
            self.board = tuple(map(tuple, rows))

    def step(self, action):
        board, gain = moved(self.board, action)
        if board == self.board:
            return False
        self.board = board; self.score += gain; self.steps += 1
        self.spawn()
        return True

    @property
    def legal_actions(self):
        return [a for a in ACTIONS if moved(self.board, a)[0] != self.board]

    @property
    def terminal(self):
        return not self.legal_actions

    @property
    def max_tile(self):
        return max(map(max, self.board))

    def state(self):
        return dict(board=self.board, score=self.score, steps=self.steps)
