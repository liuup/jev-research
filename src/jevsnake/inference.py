"""Batched probability inference and observed-event metrics."""

from contextlib import contextmanager

import numpy as np
import torch

from .batching import collate
from .events import OUTCOMES, UTILITIES


@contextmanager
def inference_precision(precision):
    if precision not in ("fp32", "bf16"):
        raise ValueError("inference precision must be fp32 or bf16")
    matmul = torch.backends.cuda.matmul.allow_tf32
    cudnn = torch.backends.cudnn.allow_tf32
    try:
        if precision == "fp32":
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = matmul
        torch.backends.cudnn.allow_tf32 = cudnn


@torch.no_grad()
def predict(model, tokenizer, rows, microbatch=64, max_length=512, precision="bf16"):
    if not rows:
        return np.empty((0, len(OUTCOMES)))
    model.eval()
    device = next(model.parameters()).device
    predictions = []
    with inference_precision(precision):
        for offset in range(0, len(rows), microbatch):
            batch = collate(rows[offset : offset + microbatch], tokenizer, max_length)
            with torch.autocast(
                device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda" and precision == "bf16",
            ):
                logp = model(batch)
            predictions.extend(logp.exp().cpu().tolist())
    return np.asarray(predictions)


def reliability(probabilities, observed, bins=10):
    indexes = np.minimum((probabilities * bins).astype(int), bins - 1)
    rows, ece = [], 0.0
    for index in range(bins):
        selected = indexes == index
        count = int(selected.sum())
        confidence = float(probabilities[selected].mean()) if count else None
        frequency = float(observed[selected].mean()) if count else None
        if count:
            ece += count / len(probabilities) * abs(confidence - frequency)
        rows.append(
            {
                "lower": index / bins,
                "upper": (index + 1) / bins,
                "count": count,
                "probability": confidence,
                "frequency": frequency,
            }
        )
    return float(ece), rows


def observed_metrics(probabilities, rows):
    targets = np.asarray([OUTCOMES.index(row["observed_outcome"]) for row in rows])
    onehot = np.eye(len(OUTCOMES))[targets]
    expected = probabilities @ UTILITIES
    observed_utility = UTILITIES[targets]
    metrics = {
        "events": len(rows),
        "top1_accuracy": float((probabilities.argmax(-1) == targets).mean()),
        "observed_nll": float(
            -np.log(np.maximum(probabilities[np.arange(len(targets)), targets], 1e-30)).mean()
        ),
        "observed_brier": float(((probabilities - onehot) ** 2).sum(-1).mean()),
        "expected_utility_mean": float(expected.mean()),
        "observed_utility_mean": float(observed_utility.mean()),
        "utility_mae": float(np.abs(expected - observed_utility).mean()),
        "utility_mse": float(((expected - observed_utility) ** 2).mean()),
        "class_counts": {
            outcome: int((targets == index).sum())
            for index, outcome in enumerate(OUTCOMES)
        },
        "calibration": {},
    }
    for index, outcome in enumerate(OUTCOMES):
        ece, diagram = reliability(probabilities[:, index], (targets == index).astype(float))
        metrics["calibration"][outcome] = {"ece": ece, "reliability": diagram}
    metrics["mean_class_ece"] = float(
        np.mean([metrics["calibration"][outcome]["ece"] for outcome in OUTCOMES])
    )
    return metrics
