"""Step-wise arithmetic reasoning environment; the calculator performs every operation.

A state holds the quantities known so far. The model picks one structured candidate:

* ``COMBINE:<op>:<i>:<j>`` combines two known quantities with a calculator operation.
  ADD and MUL are offered once per unordered pair, SUB and DIV once per ordered pair so
  both operand orders are reachable.
* ``STOP:<i>`` reports quantity ``i`` as the final answer.

Combining consumes both operands and appends the result, so the pool shrinks by one and
an episode started from ``n`` numbers has at most ``n - 1`` combine steps.
"""

from fractions import Fraction

OPS = ("ADD", "SUB", "MUL", "DIV")
COMMUTATIVE = ("ADD", "MUL")
SYMBOL = dict(ADD="+", SUB="-", MUL="*", DIV="/")


class CandidateError(ValueError):
    """Raised for malformed or unavailable candidates."""


def apply_op(op, left, right):
    if op == "ADD":
        return left + right
    if op == "SUB":
        return left - right
    if op == "MUL":
        return left * right
    if op == "DIV":
        if right == 0:
            raise ZeroDivisionError("Division by zero")
        return left / right
    raise CandidateError(f"Unknown operation: {op}")


class State:
    def __init__(self, row, quantities, steps=(), depth=0):
        self.row = row
        self.quantities = tuple(quantities)
        self.steps = tuple(steps)
        self.depth = depth

    @classmethod
    def initial(cls, row):
        return cls(row, row["numbers"])

    @property
    def question(self):
        return self.row["question"]

    @property
    def gold(self):
        return self.row["gold"]

    @property
    def values(self):
        return tuple(value for value, _ in self.quantities)

    @property
    def labels(self):
        return tuple(label for _, label in self.quantities)

    def state_id(self):
        return f"{self.row['id']}:step{self.depth}"

    def describe(self, candidate):
        """Candidate text that contains the calculator result of the operation."""
        if candidate["kind"] == "STOP":
            index = candidate["i"]
            return f"STOP: report {self.labels[index]} = {self.quantities[index][0]}"
        left_value, left_label = self.quantities[candidate["i"]]
        right_value, right_label = self.quantities[candidate["j"]]
        return (
            f"{candidate['op']}: {left_label}({left_value}) {SYMBOL[candidate['op']]} "
            f"{right_label}({right_value}) = {candidate['value']}"
        )

    def candidates(self):
        result = []
        size = len(self.quantities)
        for i in range(size):
            for j in range(size):
                if i == j:
                    continue
                for op in OPS:
                    if op in COMMUTATIVE and j <= i:
                        continue
                    if op == "DIV" and self.quantities[j][0] == 0:
                        continue
                    result.append(
                        dict(
                            kind="COMBINE",
                            op=op,
                            i=i,
                            j=j,
                            value=apply_op(
                                op, self.quantities[i][0], self.quantities[j][0]
                            ),
                            key=f"COMBINE:{op}:{i}:{j}",
                        )
                    )
        for i in range(size):
            result.append(
                dict(
                    kind="STOP",
                    op="STOP",
                    i=i,
                    j=-1,
                    value=self.quantities[i][0],
                    key=f"STOP:{i}",
                )
            )
        return result

    def combine(self, candidate):
        if candidate["kind"] != "COMBINE":
            raise CandidateError("combine() needs a COMBINE candidate")
        left_value, left_label = self.quantities[candidate["i"]]
        right_value, right_label = self.quantities[candidate["j"]]
        label = f"({left_label}{SYMBOL[candidate['op']]}{right_label})"
        kept = [
            item
            for index, item in enumerate(self.quantities)
            if index not in (candidate["i"], candidate["j"])
        ]
        kept.append((candidate["value"], label))
        step = dict(
            depth=self.depth + 1,
            key=candidate["key"],
            op=candidate["op"],
            left=str(left_value),
            right=str(right_value),
            result=str(candidate["value"]),
        )
        return State(self.row, kept, self.steps + (step,), self.depth + 1)

    def is_correct(self, value):
        return value == self.gold


def forced_stop_value(state):
    """Deterministic answer when the pool collapses or the step limit is reached."""
    return state.quantities[0][0] if len(state.quantities) == 1 else state.quantities[-1][0]


def is_final_quantity(value):
    return isinstance(value, Fraction)
