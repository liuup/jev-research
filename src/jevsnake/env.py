"""Deterministic Snake dynamics with a reproducible food stream.

The head-first state representation, tail-vacancy rule, and SplitMix64 food
stream are adapted from TianyuCodings/NanoJev (MIT license).  The simulator is
kept independent of PyTorch so environment tests run on CPU.
"""

import copy

DIRECTIONS = {
    "north": (-1, 0),
    "east": (0, 1),
    "south": (1, 0),
    "west": (0, -1),
}
REVERSE = {
    "north": "south",
    "south": "north",
    "east": "west",
    "west": "east",
}
MASK64 = (1 << 64) - 1


def _random64(state):
    state = (state + 0x9E3779B97F4A7C15) & MASK64
    value = state
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
    return state, value ^ (value >> 31)


class SnakeGame:
    def __init__(self, seed=25, size=8, initial_length=3):
        if not isinstance(size, int) or size < 2:
            raise ValueError("size must be an integer >= 2")
        if not isinstance(initial_length, int) or not 1 <= initial_length <= size:
            raise ValueError("initial_length must be in [1, size]")
        if not isinstance(seed, int):
            raise ValueError("seed must be an integer")
        self.seed = seed
        self.size = size
        self.rng_state = seed & MASK64
        row = size // 2
        head_col = max(initial_length - 1, size // 2)
        self.body = [(row, head_col - i) for i in range(initial_length)]
        self.direction = "east"
        self.score = 0
        self.steps = 0
        self.terminal = False
        self.outcome = None
        self.food = self._new_food()
        if self.food is None:
            self.terminal, self.outcome = True, "win"
        self.validate()

    @classmethod
    def from_state(cls, state):
        required = {
            "seed",
            "size",
            "rng_state",
            "body",
            "direction",
            "food",
            "score",
            "steps",
            "terminal",
            "outcome",
        }
        if not isinstance(state, dict) or not required <= set(state):
            raise ValueError("Incomplete Snake state")
        game = cls.__new__(cls)
        game.seed = state["seed"]
        game.size = state["size"]
        game.rng_state = state["rng_state"]
        game.body = [tuple(cell) for cell in state["body"]]
        game.direction = state["direction"]
        game.food = None if state["food"] is None else tuple(state["food"])
        game.score = state["score"]
        game.steps = state["steps"]
        game.terminal = state["terminal"]
        game.outcome = state["outcome"]
        game.validate()
        return game

    def clone(self):
        return copy.deepcopy(self)

    @property
    def legal_actions(self):
        if self.terminal:
            return []
        return [
            action for action in DIRECTIONS if action != REVERSE[self.direction]
        ]

    def destination(self, action):
        if action not in self.legal_actions:
            raise ValueError("Action must be a legal non-reverse move")
        dr, dc = DIRECTIONS[action]
        return self.body[0][0] + dr, self.body[0][1] + dc

    def collision(self, destination, growing):
        if any(value < 0 or value >= self.size for value in destination):
            return "wall_collision"
        occupied = self.body if growing else self.body[:-1]
        return "self_collision" if destination in occupied else None

    def one_step_safe(self, action):
        destination = self.destination(action)
        return self.collision(destination, destination == self.food) is None

    def step(self, action):
        destination = self.destination(action)
        growing = destination == self.food
        collision = self.collision(destination, growing)
        self.direction = action
        self.steps += 1
        if collision:
            self.terminal = True
            self.outcome = collision
            self.validate()
            return {"ate_food": False, "collision": True, "win": False}
        self.body = [destination] + (self.body if growing else self.body[:-1])
        if growing:
            self.score += 1
            self.food = self._new_food()
            if self.food is None:
                self.terminal = True
                self.outcome = "win"
        self.validate()
        return {
            "ate_food": growing,
            "collision": False,
            "win": self.outcome == "win",
        }

    def _new_food(self):
        occupied = set(self.body)
        free = [
            (row, col)
            for row in range(self.size)
            for col in range(self.size)
            if (row, col) not in occupied
        ]
        if not free:
            return None
        limit = (1 << 64) - ((1 << 64) % len(free))
        while True:
            self.rng_state, value = _random64(self.rng_state)
            if value < limit:
                return free[value % len(free)]

    def public_state(self):
        """Visible model state: no seed, food RNG, oracle labels, or future data."""
        return {
            "game": "snake",
            "size": self.size,
            "body": [list(cell) for cell in self.body],
            "direction": self.direction,
            "food": None if self.food is None else list(self.food),
            "score": self.score,
            "steps": self.steps,
        }

    def full_state(self):
        return self.public_state() | {
            "seed": self.seed,
            "rng_state": self.rng_state,
            "terminal": self.terminal,
            "outcome": self.outcome,
        }

    def validate(self):
        if not isinstance(self.size, int) or self.size < 2:
            raise ValueError("Invalid board size")
        if not isinstance(self.rng_state, int) or not 0 <= self.rng_state <= MASK64:
            raise ValueError("Invalid RNG state")
        if self.direction not in DIRECTIONS:
            raise ValueError("Invalid direction")
        if not self.body or len(set(self.body)) != len(self.body):
            raise ValueError("Body cells must be nonempty and unique")
        for cell in self.body:
            if len(cell) != 2 or any(not 0 <= value < self.size for value in cell):
                raise ValueError("Body cell outside board")
        if any(
            abs(a[0] - b[0]) + abs(a[1] - b[1]) != 1
            for a, b in zip(self.body, self.body[1:])
        ):
            raise ValueError("Body cells must be adjacent")
        if not self.terminal and len(self.body) > 1:
            dr = self.body[0][0] - self.body[1][0]
            dc = self.body[0][1] - self.body[1][1]
            if (dr, dc) != DIRECTIONS[self.direction]:
                raise ValueError("Direction contradicts head-neck orientation")
        occupied = set(self.body)
        if self.food is not None and (
            self.food in occupied
            or any(not 0 <= value < self.size for value in self.food)
        ):
            raise ValueError("Food must be a free in-bounds cell")
        if self.terminal:
            if self.outcome not in {"win", "wall_collision", "self_collision"}:
                raise ValueError("Invalid terminal outcome")
        elif self.outcome is not None or self.food is None:
            raise ValueError("Live state must have food and no terminal outcome")
