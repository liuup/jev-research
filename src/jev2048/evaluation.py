"""Observed-event metrics and separate evaluation-only MC comparisons."""

from pathlib import Path

import numpy as np
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from .dataset import collate
from .serialization import OUTCOMES


@torch.no_grad()
def predict(model, tokenizer, rows, microbatch=2, max_length=512):
    model.eval()
    predictions = []
    for i in range(0, len(rows), microbatch):
        batch = collate(rows[i : i + microbatch], tokenizer, max_length)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lp = model(batch)
        predictions.extend(lp.exp().cpu().tolist())
    return np.array(predictions)


def reliability(p, y, bins=10):
    # Fixed [0,.1),...,[.9,1] bins. Empty bins have null means.
    idx = np.minimum((p * bins).astype(int), bins - 1)
    rows = []
    ece = 0.0
    for b in range(bins):
        take = idx == b
        n = int(take.sum())
        confidence = float(p[take].mean()) if n else None
        frequency = float(y[take].mean()) if n else None
        if n:
            ece += n / len(p) * abs(confidence - frequency)
        rows.append(
            dict(
                lower=b / bins,
                upper=(b + 1) / bins,
                count=n,
                probability=confidence,
                frequency=frequency,
            )
        )
    return float(ece), rows


def observed_metrics(p, rows):
    y = np.array([OUTCOMES.index(r["observed_outcome"]) for r in rows])
    onehot = np.eye(len(OUTCOMES))[y]
    result = dict(
        top1_accuracy=float((p.argmax(-1) == y).mean()),
        observed_nll=float(-np.log(np.maximum(p[np.arange(len(y)), y], 1e-30)).mean()),
        observed_brier=float(((p - onehot) ** 2).sum(-1).mean()),
        events={},
    )
    for threshold in (1024, 2048, 4096):
        start = OUTCOMES.index(str(threshold))
        probability = p[:, start:].sum(-1)
        event = (y >= start).astype(float)
        ece, diagram = reliability(probability, event)
        result[f"ece_ge{threshold}"] = ece
        result["events"][str(threshold)] = dict(
            brier=float(((probability - event) ** 2).mean()),
            ece=ece,
            reliability=diagram,
        )
    return result


def mc_metrics(p, rows):
    q = np.array([r["q"] for r in rows])
    mid = (p + q) / 2

    def kl(a, b):
        return (a * (np.log(np.maximum(a, 1e-30)) - np.log(np.maximum(b, 1e-30)))).sum(
            -1
        )

    return dict(
        mc_squared_l2=float(((p - q) ** 2).sum(-1).mean()),
        mc_mae=float(abs(p - q).mean()),
        mc_js=float((0.5 * kl(p, mid) + 0.5 * kl(q, mid)).mean()),
        mc_sampling_squared_l2_floor=float(
            np.mean(
                [
                    (1 - (np.array(r["q"]) ** 2).sum()) / (r["rollouts"] - 1)
                    for r in rows
                ]
            )
        ),
    )


def plot_reliability(metrics, name):
    Path("results/plots").mkdir(parents=True, exist_ok=True)
    fig = Figure(figsize=(12, 4))
    FigureCanvasAgg(fig)
    axes = fig.subplots(1, 3)
    for ax, (threshold, event) in zip(axes, metrics["events"].items()):
        rows = [r for r in event["reliability"] if r["count"]]
        ax.plot([0, 1], [0, 1], "--", color="gray")
        ax.plot([r["probability"] for r in rows], [r["frequency"] for r in rows], "o-")
        ax.set(
            xlim=(0, 1),
            ylim=(0, 1),
            xlabel="Predicted probability",
            ylabel="Observed frequency",
            title=f"Maximum tile >= {threshold}",
        )
    fig.tight_layout()
    fig.savefig(f"results/plots/{name}_reliability.png", dpi=150)
