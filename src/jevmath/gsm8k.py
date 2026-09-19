"""GSM8K loading, answer parsing and the train/validation split."""

import json
import random
import re
from fractions import Fraction
from pathlib import Path

from .utils import digest

BASE_URL = (
    "https://raw.githubusercontent.com/openai/grade-school-math/master/"
    "grade_school_math/data"
)
TRAIN_URL = f"{BASE_URL}/train.jsonl"
TEST_URL = f"{BASE_URL}/test.jsonl"

NUMBER = re.compile(r"\d+(?:,\d{3})*(?:\.\d+)?")
INVALID = re.compile(r"[\d,]*\.?\d+")
ORDINAL_WORDS = ("first", "second", "third")


def parse_number(text):
    """Fraction form of one GSM8K number token; commas and decimals are accepted."""
    cleaned = text.replace(",", "")
    if "." in cleaned:
        return Fraction(cleaned)
    return Fraction(int(cleaned))


def final_answer(solution):
    """The ``#### N`` value of a GSM8K solution string."""
    marker = solution.rfind("####")
    if marker < 0:
        raise ValueError("Solution has no #### answer marker")
    return parse_number(solution[marker + 4 :].strip())


def question_numbers(question):
    """Given quantities in question order, as (value, label) pairs."""
    values = [parse_number(match.group(0)) for match in NUMBER.finditer(question)]
    return tuple(
        (value, f"q{index + 1}") for index, value in enumerate(values)
    )


def load_rows(path):
    rows = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        question = record["question"]
        numbers = question_numbers(question)
        if not numbers:
            continue
        rows.append(
            dict(
                id=record.get("id"),
                question=question,
                gold=final_answer(record["answer"]),
                numbers=numbers,
            )
        )
    return rows


def split_rows(rows, validation_fraction, seed):
    """Deterministic 90/10 style split of the training pool only."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must lie in (0, 1)")
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    cut = round(len(shuffled) * validation_fraction)
    if cut < 1 or cut >= len(shuffled):
        raise ValueError("Split leaves one side empty")
    return shuffled[cut:], shuffled[:cut]


def manifest(path, urls=None):
    root = Path(path)
    return dict(
        train_sha256=digest(root / "train.jsonl"),
        test_sha256=digest(root / "test.jsonl"),
        urls=urls or dict(train=TRAIN_URL, test=TEST_URL),
    )
