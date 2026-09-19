import numpy as np
import pytest

from jev2048.evaluation import mc_metrics, observed_metrics, reliability
from jev2048.serialization import OUTCOMES


def test_perfect_predictions():
    p = np.eye(7)
    rows = [
        dict(observed_outcome=o, q=p[i].tolist(), rollouts=256)
        for i, o in enumerate(OUTCOMES)
    ]
    result = observed_metrics(p, rows)
    assert (
        result["top1_accuracy"] == 1
        and result["observed_nll"] == 0
        and result["observed_brier"] == 0
    )
    assert result["ece_ge2048"] == 0
    assert result["expected_log_tile_mean_bias"] == 0
    assert result["expected_log_tile_mean_absolute_bias"] == 0
    assert "observed_log_tile_mae" not in result
    assert "observed_log_tile_mse" not in result
    mc = mc_metrics(p, rows)
    assert mc["mc_squared_l2"] == 0
    assert mc["mc_js"] == 0
    assert mc["mc_expected_log_tile_mae"] == 0


def test_observed_metrics_report_mean_bias_not_single_event_utility_error():
    p = np.full((2, 7), 1 / 7)
    rows = [dict(observed_outcome="512"), dict(observed_outcome="1024")]
    result = observed_metrics(p, rows)
    assert result["predicted_expected_log_tile"] == pytest.approx(10)
    assert result["observed_mean_log_tile"] == pytest.approx(9.5)
    assert result["expected_log_tile_mean_bias"] == pytest.approx(0.5)
    assert result["expected_log_tile_mean_absolute_bias"] == pytest.approx(0.5)


def test_bin_boundaries():
    ece, rows = reliability(np.array([0.0, 0.1, 1.0]), np.array([0.0, 1.0, 1.0]))
    assert [rows[i]["count"] for i in (0, 1, 9)] == [1, 1, 1]
    assert ece == pytest.approx(0.3)
