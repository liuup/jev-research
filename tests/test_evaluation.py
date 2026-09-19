import numpy as np
import pytest

from jevmath.evaluation import (
    action_ranking,
    observed_metrics,
    reliability,
    risk_coverage,
)


def rows_from(events):
    return [dict(candidate_ids=["yes", "no"], observed_outcome=event) for event in events]


def test_perfect_predictions_score_zero():
    predictions = np.array([[1.0, 0.0], [0.0, 1.0]])
    result = observed_metrics(predictions, rows_from(["yes", "no"]))
    assert result["top1_accuracy"] == 1
    assert result["observed_nll"] == pytest.approx(0)
    assert result["observed_brier"] == pytest.approx(0)
    assert result["binary_brier"] == pytest.approx(0)
    assert result["ece_ge_correct"] == pytest.approx(0)
    assert result["mean_entropy"] == pytest.approx(0)


def test_vector_brier_is_twice_the_scalar_brier():
    predictions = np.array([[0.8, 0.2], [0.3, 0.7]])
    result = observed_metrics(predictions, rows_from(["yes", "no"]))
    assert result["observed_brier"] == pytest.approx(2 * result["binary_brier"])
    assert result["predicted_success_probability"] == pytest.approx(0.55)
    assert result["observed_success_rate"] == pytest.approx(0.5)


def test_reliability_bins_and_ece():
    probability = np.array([0.05, 0.15, 0.95])
    event = np.array([0.0, 0.0, 1.0])
    ece, diagram = reliability(probability, event)
    occupied = [row for row in diagram if row["count"]]
    assert len(occupied) == 3
    assert occupied[0]["probability"] == pytest.approx(0.05)
    assert occupied[0]["frequency"] == pytest.approx(0.0)
    assert ece == pytest.approx((0.05 + 0.15 + 0.05) / 3)


def test_risk_coverage_prefers_confident_rows():
    predictions = np.array([[0.99, 0.01], [0.01, 0.99], [0.5, 0.5], [0.6, 0.4]])
    rows = rows_from(["yes", "no", "yes", "no"])
    result = risk_coverage(predictions, rows, points=4)
    assert result["curve"][0]["coverage"] == pytest.approx(0.25)
    assert result["curve"][0]["accuracy"] == pytest.approx(1.0)
    assert result["curve"][-1]["accuracy"] == pytest.approx(0.5)
    assert result["mean_confidence"] > 0.5
    assert result["aurc"] > 0


def test_action_ranking_agreement_and_regret():
    model = [{"a": 0.9, "b": 0.1}, {"a": 0.2, "b": 0.8}]
    reference = [{"a": 0.9, "b": 0.1}, {"a": 0.9, "b": 0.1}]
    result = action_ranking(model, reference)
    assert result["states"] == 2
    assert result["action_agreement"] == pytest.approx(0.5)
    assert result["action_regret"] == pytest.approx(0.4)
    with pytest.raises(ValueError):
        action_ranking(model, reference[:1])
