"""Probability metrics, risk-coverage and step-level action agreement."""

from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from .dataset import collate
from .serialization import candidates


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
def predict_rows(model, tokenizer, rows, microbatch=16, max_length=1024, precision="fp32"):
    """Success distribution of every candidate path, one row per question."""
    model.eval()
    device = next(model.parameters()).device
    if precision == "fp32" and any(
        parameter.is_floating_point() and parameter.dtype != torch.float32
        for parameter in model.parameters()
    ):
        raise ValueError("FP32 inference requires FP32 model parameters")
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
    return np.array(predictions)


def outcome_correct(predictions, rows):
    """Boolean label of each row and its predicted success probability."""
    yes = np.array(
        [predictions[i, candidates(row).index("yes")] for i, row in enumerate(rows)]
    )
    event = np.array([row["observed_outcome"] == "yes" for row in rows], dtype=float)
    return yes, event


def reliability(probability, event, bins=10):
    index = np.minimum((probability * bins).astype(int), bins - 1)
    ece = 0.0
    rows = []
    for bucket in range(bins):
        take = index == bucket
        count = int(take.sum())
        confidence = float(probability[take].mean()) if count else None
        frequency = float(event[take].mean()) if count else None
        if count:
            ece += count / len(probability) * abs(confidence - frequency)
        rows.append(
            dict(
                lower=bucket / bins,
                upper=(bucket + 1) / bins,
                count=count,
                probability=confidence,
                frequency=frequency,
            )
        )
    return float(ece), rows


def observed_metrics(predictions, rows):
    """Every row needs exactly one observed outcome; MC probabilities never enter."""
    if not rows:
        raise ValueError("Empty evaluation batch")
    yes, event = outcome_correct(predictions, rows)
    observed_index = (1 - event).astype(int)  # candidate order is yes, no
    one_hot = np.eye(2)[observed_index]
    ece, diagram = reliability(yes, event)
    entropy = -(
        yes * np.log(np.maximum(yes, 1e-30))
        + (1 - yes) * np.log(np.maximum(1 - yes, 1e-30))
    )
    return dict(
        questions=len(rows),
        top1_accuracy=float((predictions.argmax(-1) == observed_index).mean()),
        observed_nll=float(
            -(
                event * np.log(np.maximum(yes, 1e-30))
                + (1 - event) * np.log(np.maximum(1 - yes, 1e-30))
            ).mean()
        ),
        observed_brier=float(((predictions - one_hot) ** 2).sum(-1).mean()),
        binary_brier=float(((yes - event) ** 2).mean()),
        predicted_success_probability=float(yes.mean()),
        observed_success_rate=float(event.mean()),
        mean_entropy=float(entropy.mean()),
        ece_ge_correct=ece,
        events={"correct": dict(brier=float(((yes - event) ** 2).mean()), ece=ece, reliability=diagram)},
    )


def risk_coverage(predictions, rows, points=10):
    """Accuracy among the most confident fraction of questions."""
    yes, event = outcome_correct(predictions, rows)
    predicted = (yes > 0.5).astype(int)
    correct = (predicted == event.astype(int)).astype(float)
    confidence = np.maximum(yes, 1 - yes)
    order = np.argsort(-confidence)
    curve = []
    for step in range(1, points + 1):
        take = order[: max(1, round(len(order) * step / points))]
        curve.append(
            dict(coverage=step / points, accuracy=float(correct[take].mean()))
        )
    risk = 1 - np.array([point["accuracy"] for point in curve])
    return dict(
        curve=curve,
        aurc=float(np.trapezoid(risk, dx=1 / points)),
        mean_confidence=float(confidence.mean()),
    )


def action_ranking(model_probabilities, reference_probabilities):
    """Step/action agreement between the model and reference success probabilities.

    Both inputs are ``{action_key: probability}`` per state. The reference is the
    empirical success rate of an action measured by repeated rollouts.
    """
    if len(model_probabilities) != len(reference_probabilities):
        raise ValueError("Model and reference need the same number of states")
    agreement, regrets, gaps = [], [], []
    for model, reference in zip(model_probabilities, reference_probabilities):
        shared = sorted(set(model) & set(reference))
        if len(shared) < 2:
            continue
        best = max(shared, key=reference.__getitem__)
        chosen = max(shared, key=model.__getitem__)
        agreement.append(best == chosen)
        regrets.append(reference[best] - reference[chosen])
        gaps.append(max(model.values()) - min(model.values()))
    return dict(
        states=len(agreement),
        action_agreement=float(np.mean(agreement)) if agreement else None,
        action_regret=float(np.mean(regrets)) if regrets else None,
        mean_model_action_gap=float(np.mean(gaps)) if gaps else None,
    )


def plot_reliability(metrics, name):
    Path("results/plots").mkdir(parents=True, exist_ok=True)
    figure = Figure(figsize=(5, 4))
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, 1)
    rows = [row for row in metrics["events"]["correct"]["reliability"] if row["count"]]
    axes.plot([0, 1], [0, 1], "--", color="gray")
    axes.plot([row["probability"] for row in rows], [row["frequency"] for row in rows], "o-")
    axes.set(
        xlim=(0, 1),
        ylim=(0, 1),
        xlabel="Predicted success probability",
        ylabel="Observed success frequency",
        title="Frozen-policy success event",
    )
    figure.tight_layout()
    figure.savefig(f"results/plots/{name}_reliability.png", dpi=150)
