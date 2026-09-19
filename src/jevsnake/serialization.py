"""Versioned text encoding for conditional Snake event distributions."""

import json

from .env import DIRECTIONS
from .events import DESCRIPTIONS, EVENT_SCHEMA, OUTCOMES


def serialize(row, candidate_id):
    if candidate_id not in OUTCOMES:
        raise ValueError(f"Unknown outcome candidate: {candidate_id}")
    if row["action"] not in DIRECTIONS:
        raise ValueError("Unknown Snake action")
    body = json.dumps(row["body"], separators=(",", ":"))
    food = json.dumps(row["food"], separators=(",", ":"))
    head = row["body"][0]
    dr, dc = DIRECTIONS[row["action"]]
    destination = [head[0] + dr, head[1] + dc]
    horizon = row["event_horizon"]
    return (
        f"[STATE]\nGame: Snake\nBoardSize: {row['size']}x{row['size']}\n"
        f"BodyHeadFirst: {body}\nDirection: {row['direction']}\nFood: {food}\n"
        f"Score: {row['score']}\n\n[ACTION]\nMove {row['action']}\n"
        f"NextHead: {json.dumps(destination, separators=(',', ':'))}\n\n"
        f"[EVENT]\nSchema: {EVENT_SCHEMA}\nExecute the action now, then follow the "
        f"frozen continuation policy for at most {horizon - 1} additional moves. "
        "Stop at the first food collection or collision.\n\n"
        "[QUESTION]\nWhich event happens first?\n\n"
        f"[CANDIDATE]\n{DESCRIPTIONS[candidate_id]}"
    )
