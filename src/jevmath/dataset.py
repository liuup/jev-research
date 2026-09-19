"""Batch assembly: one question per (state, candidate action) with its yes/no paths."""

import torch

from .serialization import candidates, serialize


def collate(rows, tokenizer, max_length=1024):
    ids = [candidates(row) for row in rows]
    if any(len(x) != len(set(x)) for x in ids):
        raise ValueError("Duplicate candidates")
    paths = [serialize(row | dict(candidate_id=value)) for row in rows for value in ids[0]]
    tokens = tokenizer(paths, padding=True, truncation=False, return_tensors="pt")
    if tokens["input_ids"].shape[1] > max_length:
        raise ValueError("Input exceeds max_length; refusing to truncate a candidate")
    offsets = [0]
    for _ in rows:
        offsets.append(offsets[-1] + len(ids[0]))
    mask = (
        torch.arange(max(map(len, ids)))[None, :]
        < torch.tensor(list(map(len, ids)))[:, None]
    )
    targets = torch.tensor(
        [
            ids[index].index(row["observed_outcome"])
            if "observed_outcome" in row
            else -1
            for index, row in enumerate(rows)
        ]
    )
    return dict(
        tokens=tokens,
        offsets=offsets,
        mask=mask,
        candidate_ids=ids[0],
        state_ids=[row["state_id"] for row in rows],
        action_keys=[row["action_key"] for row in rows],
        targets=targets,
    )
