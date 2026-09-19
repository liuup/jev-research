"""24 game environment: exact rational states, four operations, no stop action.

A state is the multiset of values still unused. Every action picks two of them and
applies one operation to the ordered pair, so both operand orders of ``-`` and ``/`` are
available. Actions that only differ by swapping the operands of ``+`` or ``*``, or by
using two copies of an equal value, collapse to one action. Dividing by zero is never
offered. An episode ends after three operations, when one value is left.
"""

from fractions import Fraction

from .puzzles import TARGET, apply_op, is_solvable

OPS = ("+", "-", "*", "/")
COMMUTATIVE = ("+", "*")
OPERATIONS_PER_EPISODE = 3


class ActionError(ValueError):
    """Raised for malformed or unavailable actions."""


class State:
    """Remaining values plus the operations already applied."""

    def __init__(self, puzzle, values, history=(), depth=0):
        self.puzzle = puzzle
        self.values = tuple(values)
        self.history = tuple(history)
        self.depth = depth

    @classmethod
    def initial(cls, puzzle):
        return cls(puzzle, tuple(sorted(Fraction(number) for number in puzzle["numbers"])))

    @property
    def puzzle_id(self):
        return self.puzzle["puzzle_id"]

    @property
    def target(self):
        return Fraction(self.puzzle["target"])

    @property
    def finished(self):
        return len(self.values) == 1

    @property
    def current(self):
        if not self.finished:
            raise ActionError("The episode needs three operations before it has a value")
        return self.values[0]

    @property
    def solved(self):
        return self.finished and self.current == self.target

    @property
    def remaining_text(self):
        return " ".join(str(value) for value in self.values)

    def state_id(self):
        return f"{self.puzzle_id}:{self.depth}:{'|'.join(str(value) for value in self.values)}"

    def actions(self):
        """Every distinct action of this state, with the exact result it produces."""
        found = []
        seen = set()
        for i in range(len(self.values)):
            for j in range(i + 1, len(self.values)):
                left, right = self.values[i], self.values[j]
                for op, a, b in (
                    ("+", left, right),
                    ("*", left, right),
                    ("-", left, right),
                    ("-", right, left),
                    ("/", left, right),
                    ("/", right, left),
                ):
                    key = action_key(op, a, b)
                    if key in seen:
                        continue
                    if op == "/" and b == 0:
                        continue
                    seen.add(key)
                    found.append(
                        dict(
                            key=key,
                            op=op,
                            left=a,
                            right=b,
                            value=apply_op(op, a, b),
                            remaining=self.remaining_after(a, b, apply_op(op, a, b)),
                        )
                    )
        return found

    def remaining_after(self, left, right, result):
        rest = list(self.values)
        rest.remove(left)
        rest.remove(right)
        rest.append(result)
        return tuple(sorted(rest))

    def apply(self, action):
        """Consume both operands and keep the result; the state stays a sorted multiset."""
        remaining = self.remaining_after(action["left"], action["right"], action["value"])
        step = dict(
            depth=self.depth + 1,
            key=action["key"],
            op=action["op"],
            left=str(action["left"]),
            right=str(action["right"]),
            result=str(action["value"]),
            text=action_text(action),
        )
        return State(self.puzzle, remaining, self.history + (step,), self.depth + 1)

    def solvable_actions(self):
        """Keys of the actions whose remaining values can still reach the target."""
        return [
            action["key"]
            for action in self.actions()
            if is_solvable(action["remaining"], self.target)
        ]


def action_key(op, left, right):
    """Stable action id; commutative operations use one operand order."""
    if op in COMMUTATIVE:
        left, right = min(left, right), max(left, right)
    return f"{op}:{left}:{right}"


def action_text(action):
    return f"{action['left']} {action['op']} {action['right']} = {action['value']}"


def split_action_key(key):
    op, left, right = key.split(":")
    return op, Fraction(left), Fraction(right)


def solved_after(state, action):
    """Whether the operation keeps the remaining values solvable."""
    return is_solvable(
        state.remaining_after(action["left"], action["right"], action["value"]), TARGET
    )
