import numpy as np
import pytest
import torch

from jev2048 import controller as controller_module
from jev2048.controller import HeuristicJudgeController, analyze_reach_2048
from jev2048.dataset import rollout_event
from jev2048.env import Game, moved
from jev2048.evaluation import observed_metrics, mc_metrics
from jev2048.heuristic import HeuristicPolicy
from jev2048.online_evaluation import play_seeded
from jev2048.policies import FrozenHeuristic, FrozenHybridPolicy
from jev2048.utils import digest


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


JUDGE_BOARD = ((32, 8, 2, 2), (8, 2, 0, 0), (2, 0, 0, 0), (0, 0, 0, 0))
SINGLE_ACTION_BOARD = (
    (2, 4, 8, 16),
    (4, 8, 16, 32),
    (8, 16, 32, 64),
    (0, 0, 0, 0),
)


def judge_game():
    return Game(0, board=JUDGE_BOARD, score=140)


def uniform_predict(calls):
    def fake(model, tokenizer, rows, microbatch, max_length, precision="fp32"):
        calls.append([row["action"] for row in rows])
        return np.full((len(rows), 2), 0.5)

    return fake


def prefers_right(model, tokenizer, rows, microbatch, max_length, precision="fp32"):
    out = np.zeros((len(rows), 2))
    for index, row in enumerate(rows):
        yes = 0.9 if row["action"] == "RIGHT" else 0.05
        out[index] = [yes, 1 - yes]
    return out


def test_judge_delta_zero_keeps_every_heuristic_decision(monkeypatch):
    calls = []
    monkeypatch.setattr(controller_module, "predict", uniform_predict(calls))
    heuristic = HeuristicPolicy("configs/heuristic.yaml")
    game = judge_game()
    values = {a: heuristic.value(*moved(game.board, a)) for a in game.legal_actions}
    assert sum(v == max(values.values()) for v in values.values()) == 1
    judge = HeuristicJudgeController(
        None, None, heuristic, 0.0, continuation_policy_id="frozen"
    )
    assert judge.choose(game) == heuristic.choose(game)
    assert calls == []
    summary = judge.summary()
    assert summary["model_decisions"] == 0 and summary["code_decisions"] == 1
    assert summary["mean_heuristic_value_sacrifice"] == 0


def test_judge_delta_and_margin_control_model_overrides(monkeypatch):
    monkeypatch.setattr(controller_module, "predict", prefers_right)
    heuristic = HeuristicPolicy("configs/heuristic.yaml")
    assert heuristic.choose(judge_game()) == "LEFT"
    wide = HeuristicJudgeController(
        None, None, heuristic, 1e6, continuation_policy_id="frozen"
    )
    assert wide.choose(judge_game()) == "RIGHT"
    summary = wide.summary()
    assert summary["model_decisions"] == 1 and summary["model_agreement_rate"] == 0
    assert summary["mean_heuristic_value_sacrifice"] == pytest.approx(
        heuristic.value(*moved(JUDGE_BOARD, "LEFT"))
        - heuristic.value(*moved(JUDGE_BOARD, "RIGHT"))
    )
    blocked = HeuristicJudgeController(
        None, None, heuristic, 1e6, margin=10.0, continuation_policy_id="frozen"
    )
    assert blocked.choose(judge_game()) == "LEFT"
    assert blocked.summary()["margin_overrides"] == 1


def test_judge_skips_single_legal_action(monkeypatch):
    calls = []
    monkeypatch.setattr(controller_module, "predict", uniform_predict(calls))
    heuristic = HeuristicPolicy("configs/heuristic.yaml")
    game = Game(1, board=SINGLE_ACTION_BOARD)
    assert game.legal_actions == ["DOWN"]
    judge = HeuristicJudgeController(
        None, None, heuristic, 5.0, continuation_policy_id="frozen"
    )
    assert judge.choose(game) == "DOWN"
    assert calls == [] and judge.summary()["single_legal_action"] == 1


def hybrid_policy(delta, heuristic_config="configs/heuristic.yaml"):
    frozen = FrozenHeuristic(heuristic_config)
    spec = dict(
        kind="jev-hybrid",
        policy_id=f"jev-hybrid:test-delta-{delta}",
        utility="threshold",
        threshold=2048,
        inference_precision="fp32",
        task="reach_2048",
        continuation_policy_id=f"heuristic:{digest(heuristic_config)}",
        judge_delta=delta,
        judge_margin=0.0,
        heuristic_config=heuristic_config,
    )
    policy = FrozenHybridPolicy(
        torch.nn.Linear(2, 2),
        None,
        spec,
        dict(max_length=512, inference_questions=16),
    )
    return policy, frozen


def test_judge_delta_zero_reproduces_heuristic_games(monkeypatch):
    calls = []
    monkeypatch.setattr(controller_module, "predict", uniform_predict(calls))
    policy, frozen = hybrid_policy(0.0)
    seeds = [0, 1]
    judged = play_seeded(policy, seeds, 2, Game, success_tile=2048)
    baseline = play_seeded(frozen, seeds, 2, Game, success_tile=2048)
    assert judged["records"] == baseline["records"]
    assert judged["reach_2048"] == baseline["reach_2048"]
    assert policy.summary()["mean_heuristic_value_sacrifice"] == 0


def test_judge_rejects_a_foreign_continuation_policy():
    policy, _ = hybrid_policy(0.1)
    foreign = dict(policy.spec, continuation_policy_id="heuristic:not-the-label-policy")
    with pytest.raises(ValueError, match="continuation policy"):
        FrozenHybridPolicy(
            torch.nn.Linear(2, 2),
            None,
            foreign,
            dict(max_length=512, inference_questions=16),
        )
    # The frozen data contract identifies the continuation policy by the config
    # digest, not by FrozenHeuristic.policy_id.
    assert policy.spec["continuation_policy_id"] != FrozenHeuristic(
        "configs/heuristic.yaml"
    ).policy_id
