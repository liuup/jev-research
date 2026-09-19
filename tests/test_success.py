import numpy as np
import pytest

from jev2048.controller import analyze_reach_2048
from jev2048.dataset import rollout_event
from jev2048.env import Game
from jev2048.evaluation import observed_metrics, mc_metrics


def test_success_metrics():
    rows = [dict(task="reach_2048", observed_outcome=y) for y in ("yes", "no")]
    p = np.array([[.8, .2], [.3, .7]])
    m = observed_metrics(p, rows)
    assert m["binary_brier"] == pytest.approx(.065)
    assert m["observed_brier"] == pytest.approx(.13)
    assert m["top1_accuracy"] == 1
    assert "predicted_expected_log_tile" not in m
    assert mc_metrics(p, [dict(q=r) for r in p])["mc_squared_l2"] == 0


def test_rollout_stops_on_success():
    class NeverContinue:
        def choose(self, game):
            raise AssertionError("Must stop immediately at success")

    state = Game(board=((1024, 1024, 0, 0), (0, 0, 0, 0),
                        (0, 0, 0, 0), (0, 0, 0, 0))).state()
    event = rollout_event(state | dict(task="reach_2048"), "LEFT", 17, NeverContinue())
    assert event["observed_outcome"] == "yes" and event["rollout_steps"] == 1


def test_all_actions_use_one_predict_call(monkeypatch):
    calls = []

    def fake_predict(model, tokenizer, rows, **kwargs):
        calls.append((rows, kwargs))
        return np.array([[.2, .8]] * len(rows))

    monkeypatch.setattr("jev2048.controller.predict", fake_predict)
    game = Game(board=((0, 0, 0, 0), (0, 2, 2, 0), (0, 0, 0, 0), (0, 0, 0, 0)))
    result = analyze_reach_2048(None, None, game, "frozen")
    assert len(calls) == 1 and calls[0][1]["microbatch"] == 4
    assert list(result) == game.legal_actions
    assert all(v == {"yes": .2, "no": .8} for v in result.values())
