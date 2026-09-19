"""Enumerate, solve and split the 24 game hands.

A hand is four integers from 1 to 13 in non-decreasing order, so permutations are
collapsed at generation time and every hand is its own group key. All arithmetic uses
``Fraction``: intermediate results may be negative or fractional and nothing is rounded.
"""

import hashlib
import itertools
import json
import random
import subprocess
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

from .utils import digest, save_json, write_jsonl

TARGET = Fraction(24)
NUMBER_RANGE = (1, 13)
HAND_SIZE = 4
OPERATIONS = ("+", "-", "*", "/")
PRECEDENCE = {"+": 1, "-": 1, "*": 2, "/": 2}
SOLVABLE, UNSOLVABLE = 1_362, 458
SPLIT_SIZES = {"train": 954, "dev": 204, "test": 204}


def hands(low=NUMBER_RANGE[0], high=NUMBER_RANGE[1], size=HAND_SIZE):
    """Every non-decreasing hand, in enumeration order."""
    return list(
        itertools.combinations_with_replacement(range(low, high + 1), size)
    )


def canonical_key(numbers):
    """Group key that makes permutations and orderings collapse to one puzzle."""
    return tuple(sorted(Fraction(number) for number in numbers))


def puzzle_id(numbers):
    return "_".join(str(value) for value in numbers)


def apply_op(op, left, right):
    if op == "+":
        return left + right
    if op == "-":
        return left - right
    if op == "*":
        return left * right
    if op == "/":
        if right == 0:
            raise ZeroDivisionError("Division by zero")
        return left / right
    raise ValueError(f"Unknown operation: {op}")


def operations(values):
    """Every reachable operation with its operands, commutative pairs offered once."""
    for i, j in itertools.combinations(range(len(values)), 2):
        left, right = values[i], values[j]
        yield "+", left, right
        yield "*", left, right
        yield "-", left, right
        yield "-", right, left
        if right != 0:
            yield "/", left, right
        if left != 0:
            yield "/", right, left


def _remainder(values, left, right, result):
    rest = list(values)
    rest.remove(left)
    rest.remove(right)
    rest.append(result)
    return tuple(sorted(rest))


def is_solvable(values, target=TARGET, _memo=None):
    """Exact reachability of the target from a multiset of values."""
    memo = {} if _memo is None else _memo
    state = tuple(sorted(Fraction(value) for value in values))
    if state in memo:
        return memo[state]
    if len(state) == 1:
        memo[state] = state[0] == target
    else:
        memo[state] = any(
            is_solvable(_remainder(state, left, right, apply_op(op, left, right)), target, memo)
            for op, left, right in operations(state)
        )
    return memo[state]


def render(expression, parent_op=None, side=None):
    """Text for an expression tuple, with only the parentheses a reader needs."""
    if expression[0] == "num":
        return str(expression[1])
    _, op, left, right = expression
    own = PRECEDENCE[op]
    parent = PRECEDENCE.get(parent_op, 0)
    text = (
        f"{render(left, op, 'left')} {op} {render(right, op, 'right')}"
    )
    needs_parentheses = own < parent or (
        own == parent and side == "right" and parent_op in ("-", "/")
    )
    return f"({text})" if needs_parentheses else text


def canonical(expression):
    """Commutative operands in a fixed order; duplicates then render identically."""
    if expression[0] == "num":
        return expression
    _, op, left, right = expression
    left, right = canonical(left), canonical(right)
    if op in ("+", "*") and render(right) < render(left):
        left, right = right, left
    return ("op", op, left, right)


def expression_solutions(values, target=TARGET, _memo=None):
    """Every valid expression reaching the target, with each input leaf used once.

    A value-only reconstruction is ambiguous when an intermediate result equals an
    unrelated remaining input.  This subset dynamic program retains expression
    provenance, so it cannot accidentally replace multiple equal-valued leaves.
    """
    del _memo  # Kept in the signature for compatibility with older callers.
    numbers = tuple(Fraction(value) for value in values)
    by_subset = {
        1 << index: {value: {("num", value)}}
        for index, value in enumerate(numbers)
    }
    full = (1 << len(numbers)) - 1
    for mask in range(1, full + 1):
        if mask in by_subset:
            continue
        table = {}
        left_mask = (mask - 1) & mask
        while left_mask:
            right_mask = mask ^ left_mask
            if right_mask and left_mask < right_mask:
                for left_value, left_nodes in by_subset[left_mask].items():
                    for right_value, right_nodes in by_subset[right_mask].items():
                        candidates = [
                            ("+", left_value, right_value, left_nodes, right_nodes),
                            ("*", left_value, right_value, left_nodes, right_nodes),
                            ("-", left_value, right_value, left_nodes, right_nodes),
                            ("-", right_value, left_value, right_nodes, left_nodes),
                        ]
                        if right_value != 0:
                            candidates.append(
                                ("/", left_value, right_value, left_nodes, right_nodes)
                            )
                        if left_value != 0:
                            candidates.append(
                                ("/", right_value, left_value, right_nodes, left_nodes)
                            )
                        for op, left, right, left_expressions, right_expressions in candidates:
                            result = apply_op(op, left, right)
                            expressions = table.setdefault(result, set())
                            for left_expression in left_expressions:
                                for right_expression in right_expressions:
                                    expressions.add(
                                        canonical(
                                            ("op", op, left_expression, right_expression)
                                        )
                                    )
            left_mask = (left_mask - 1) & mask
        by_subset[mask] = table
    return by_subset[full].get(Fraction(target), set())


def solution_strings(values, target=TARGET):
    """Distinct rendered solutions; commutative swaps and equal numbers collapse."""
    return {render(expression) for expression in expression_solutions(values, target)}


def evaluate_expression(text):
    """Value of a rendered solution, used by the tests to check the generator."""
    tokens = text.replace("(", " ( ").replace(")", " ) ").split()
    position = 0

    def peek():
        return tokens[position] if position < len(tokens) else None

    def advance():
        nonlocal position
        token = tokens[position]
        position += 1
        return token

    def parse_expression():
        value = parse_term()
        while peek() in ("+", "-"):
            value = apply_op(advance(), value, parse_term())
        return value

    def parse_term():
        value = parse_factor()
        while peek() in ("*", "/"):
            value = apply_op(advance(), value, parse_factor())
        return value

    def parse_factor():
        token = advance()
        if token == "(":
            value = parse_expression()
            if advance() != ")":
                raise ValueError(f"Missing closing parenthesis in {text}")
            return value
        return Fraction(token)

    value = parse_expression()
    if position != len(tokens):
        raise ValueError(f"Trailing tokens in {text}")
    return value


def first_step_actions(numbers):
    """Deduplicated first-step actions, each with whether the rest stays solvable."""
    values = tuple(sorted(Fraction(number) for number in numbers))
    rows = []
    seen = set()
    for op, left, right in operations(values):
        key = (
            (op, min(left, right), max(left, right))
            if op in ("+", "*")
            else (op, left, right)
        )
        if key in seen:
            continue
        seen.add(key)
        result = apply_op(op, left, right)
        remainder = _remainder(values, left, right, result)
        rows.append(
            dict(
                action_key=action_key(op, left, right),
                op=op,
                left=str(left),
                right=str(right),
                result=str(result),
                action_text=f"{left} {op} {right} = {result}",
                remaining=[str(value) for value in remainder],
                solvable_after=is_solvable(remainder),
            )
        )
    return rows


def action_key(op, left, right):
    """Stable action id; commutative operations use a single operand order."""
    if op in ("+", "*"):
        left, right = min(left, right), max(left, right)
    return f"{op}:{left}:{right}"


def build_puzzles(low=NUMBER_RANGE[0], high=NUMBER_RANGE[1], size=HAND_SIZE):
    """Every hand with its solvability, solution count and one worked solution."""
    puzzles = []
    for hand in hands(low, high, size):
        values = tuple(Fraction(number) for number in hand)
        found = solution_strings(values) if is_solvable(values) else set()
        puzzles.append(
            dict(
                puzzle_id=puzzle_id(hand),
                numbers=list(hand),
                target=int(TARGET),
                solvable=bool(found),
                solution_count=len(found),
                example_solution=min(sorted(found), key=len) if found else None,
            )
        )
    return puzzles


def split_puzzles(puzzles, seed=17, sizes=SPLIT_SIZES):
    """Deterministic split of the solvable puzzles, keyed by canonical hand."""
    if sum(sizes.values()) != len(puzzles):
        raise ValueError(f"Split sizes {sizes} do not cover {len(puzzles)} puzzles")
    if len({canonical_key(puzzle["numbers"]) for puzzle in puzzles}) != len(puzzles):
        raise ValueError("Puzzles must be one per canonical hand")
    order = sorted(puzzle["puzzle_id"] for puzzle in puzzles)
    random.Random(seed).shuffle(order)
    by_id = {puzzle["puzzle_id"]: puzzle for puzzle in puzzles}
    cursor = 0
    splits = {}
    for name in ("train", "dev", "test"):
        end = cursor + sizes[name]
        splits[name] = [by_id[key] for key in sorted(order[cursor:end])]
        cursor = end
    return splits


def _split_hash(puzzles):
    payload = json.dumps([puzzle["puzzle_id"] for puzzle in puzzles], sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def _code_version():
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return f"{commit}+dirty" if dirty else commit
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def write_dataset(root, seed=17, low=NUMBER_RANGE[0], high=NUMBER_RANGE[1], size=HAND_SIZE):
    """Build every file, then the manifest that pins the result."""
    root = Path(root)
    puzzles = build_puzzles(low, high, size)
    solvable = [puzzle for puzzle in puzzles if puzzle["solvable"]]
    unsolvable = [puzzle for puzzle in puzzles if not puzzle["solvable"]]
    if len(puzzles) != SOLVABLE + UNSOLVABLE:
        raise ValueError(f"Expected {SOLVABLE + UNSOLVABLE} hands, found {len(puzzles)}")
    if len(solvable) != SOLVABLE:
        raise ValueError(f"Expected {SOLVABLE} solvable hands, found {len(solvable)}")
    if len(unsolvable) != UNSOLVABLE:
        raise ValueError(f"Expected {UNSOLVABLE} unsolvable hands, found {len(unsolvable)}")
    splits = split_puzzles(solvable, seed)
    oracle = [
        dict(
            puzzle_id=puzzle["puzzle_id"],
            depth=0,
            actions=first_step_actions(puzzle["numbers"]),
        )
        for puzzle in solvable
    ]
    files = {
        "train.jsonl": splits["train"],
        "dev.jsonl": splits["dev"],
        "test.jsonl": splits["test"],
        "unsolvable.jsonl": unsolvable,
        "oracle_actions.jsonl": oracle,
    }
    for name, rows in files.items():
        write_jsonl(root / name, rows)
    manifest = dict(
        task="24 game",
        target=int(TARGET),
        number_range=[low, high],
        hand_size=size,
        operations=list(OPERATIONS),
        rules=dict(
            use_every_number_once=True,
            non_decreasing_hands_only=True,
            fractions_and_negatives_allowed=True,
            exact_arithmetic="fractions.Fraction",
            stop_action=False,
            operations_per_episode=size - 1,
        ),
        seed=seed,
        split_sizes=SPLIT_SIZES,
        counts=dict(
            hands=len(puzzles),
            solvable=len(solvable),
            unsolvable=len(unsolvable),
            train=len(splits["train"]),
            dev=len(splits["dev"]),
            test=len(splits["test"]),
        ),
        split_hashes={name: _split_hash(rows) for name, rows in splits.items()},
        file_hashes={name: digest(root / name) for name in files},
        code_version=_code_version(),
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    save_json(root / "manifest.json", manifest)
    return manifest


def load_puzzles(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def load_oracle(root):
    """``{(puzzle_id, depth): {action_key: solvable_after}}`` for oracle scoring."""
    table = {}
    for row in load_puzzles(Path(root) / "oracle_actions.jsonl"):
        table[(row["puzzle_id"], row["depth"])] = {
            action["action_key"]: action["solvable_after"] for action in row["actions"]
        }
    return table


def verify_manifest(root, manifest=None):
    """Fail fast when the files on disk drift away from the manifest."""
    root = Path(root)
    manifest = manifest or json.loads((root / "manifest.json").read_text())
    counts = manifest["counts"]
    for name, expected in (
        ("train.jsonl", counts["train"]),
        ("dev.jsonl", counts["dev"]),
        ("test.jsonl", counts["test"]),
        ("unsolvable.jsonl", counts["unsolvable"]),
    ):
        rows = load_puzzles(root / name)
        if len(rows) != expected:
            raise ValueError(f"{name} holds {len(rows)} rows, manifest says {expected}")
        if digest(root / name) != manifest["file_hashes"][name]:
            raise ValueError(f"{name} does not match its manifest hash")
    return manifest
