"""Complete-question batching; outcome candidates are never split or truncated."""

import torch

from .events import OUTCOMES
from .serialization import serialize


def collate(rows, tokenizer, max_length=512):
    candidate_ids = [list(row.get("candidate_ids", OUTCOMES)) for row in rows]
    if not rows or any(
        ids != list(OUTCOMES) or len(ids) != len(set(ids)) for ids in candidate_ids
    ):
        raise ValueError("Every event must contain the complete versioned outcome set")
    paths = [
        serialize(row, candidate)
        for row, candidates in zip(rows, candidate_ids)
        for candidate in candidates
    ]
    tokens = tokenizer(paths, padding=True, truncation=False, return_tensors="pt")
    if tokens["input_ids"].shape[1] > max_length:
        raise ValueError("Input exceeds max_length; candidate paths are never truncated")
    offsets = [0]
    for candidates in candidate_ids:
        offsets.append(offsets[-1] + len(candidates))
    mask = torch.ones((len(rows), len(OUTCOMES)), dtype=torch.bool)
    targets = torch.tensor(
        [
            OUTCOMES.index(row["observed_outcome"])
            if "observed_outcome" in row
            else -1
            for row in rows
        ],
        dtype=torch.long,
    )
    return {
        "tokens": tokens,
        "offsets": offsets,
        "mask": mask,
        "candidate_ids": candidate_ids,
        "action_ids": [row["action"] for row in rows],
        "state_ids": [row["state_id"] for row in rows],
        "targets": targets,
    }
