import copy
import json

import numpy as np
import pytest
import torch
from torch import nn

from jev2048 import controller as controller_module
from jev2048 import online_trainer
from jev2048.config import load_config
from jev2048.controller import JevController
from jev2048.env import Game
from jev2048.objectives import masked_log_probs
from jev2048.online import (
    AsyncOnlineCollector,
    OnlineCollector,
    mc_reference,
    retarget_questions,
    stream_seed,
    terminal_feedback,
    validate_events,
)
from jev2048.online_evaluation import action_ranking_metrics, play_seeded, promotion_decision
from jev2048.online_trainer import run_online, validate_online_config
from jev2048.policies import FrozenHeuristic, FrozenModelPolicy
from jev2048.serialization import DESCRIPTIONS
from jev2048.verification import audit


def test_evaluation_progress(capsys):
    context = dict(generation=0, phase="test", role="candidate")
    policy = FrozenHeuristic("configs/heuristic.yaml")
    result = play_seeded(policy, [1, 2, 3], 2, tiny_game, progress=context)
    logs = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert logs[0]["event"] == "evaluation_start"
    assert logs[-1]["event"] == "evaluation_complete"
    assert logs[-1]["completed_games"] == result["games"] == 3
    assert any(row["event"] == "evaluation_progress" for row in logs)
    assert all(row["phase"] == "test" and row["total_games"] == 3 for row in logs)
    counts = [row["completed_games"] for row in logs]
    assert counts == sorted(counts)


class ScheduledGame:
    def __init__(self, seed):
        self.seed = seed
        self.steps = self.score = 0
        self.max_tile = 2
        self.terminal = seed == 0

    def step(self, action):
        self.steps += 1
        self.score += 4
        self.terminal = self.steps >= self.seed
        return True


class RecordingPolicy:
    policy_id = "test"

    def __init__(self):
        self.batches = []

    def choose_many(self, games):
        self.batches.append([g.seed for g in games])
        return ["LEFT"] * len(games)


def test_evaluation_refills_and_preserves_order():
    policy = RecordingPolicy()
    seeds = [1, 4, 2, 0, 1]
    result = play_seeded(policy, seeds, 2, ScheduledGame)
    assert policy.batches[:2] == [[1, 4], [4, 2]]
    assert len(policy.batches) == 4
    assert [r["seed"] for r in result["records"]] == seeds
    assert [r["steps"] for r in result["records"]] == seeds
    serial = play_seeded(RecordingPolicy(), seeds, 1, ScheduledGame)
    assert result == serial
    with pytest.raises(ValueError, match="batch_size"):
        play_seeded(policy, seeds, 0, ScheduledGame)


def test_evaluation_real_environment_batch_invariance():
    policy = FrozenHeuristic("configs/heuristic.yaml")
    seeds = list(range(12))
    assert play_seeded(policy, seeds, 1, tiny_game) == play_seeded(
        policy, seeds, 4, tiny_game
    )


class SuccessGame:
    """Stub whose maximum tile doubles after every step."""

    def __init__(self, seed):
        self.seed = seed
        self.steps = self.score = 0
        self.max_tile = 2
        self.terminal = False

    def step(self, action):
        self.steps += 1
        self.max_tile *= 2
        return True


def success_board(seed):
    return Game(seed, board=((1024, 1024, 0, 0), (0,) * 4, (0,) * 4, (0,) * 4))


class LeftPolicy:
    policy_id = "left"

    def choose_many(self, games):
        return ["LEFT"] * len(games)


def test_play_seeded_stops_at_success_tile():
    result = play_seeded(LeftPolicy(), [0, 1], 1, success_board, success_tile=2048)
    assert [r["steps"] for r in result["records"]] == [1, 1]
    assert [r["max_tile"] for r in result["records"]] == [2048, 2048]
    assert result["reach_2048"] == 1.0
    assert (
        play_seeded(LeftPolicy(), [0, 1], 2, success_board, success_tile=2048)
        == result
    )
    with pytest.raises(ValueError, match="success_tile"):
        play_seeded(LeftPolicy(), [0], 1, success_board, success_tile=1)


def test_play_seeded_success_games_refill_slots():
    result = play_seeded(RecordingPolicy(), [1, 2, 3], 2, SuccessGame, success_tile=8)
    assert [r["steps"] for r in result["records"]] == [2, 2, 2]
    assert [r["max_tile"] for r in result["records"]] == [8, 8, 8]
    assert result["reach_2048"] == 0.0
    assert play_seeded(RecordingPolicy(), [1, 2, 3], 1, SuccessGame, success_tile=8) == result


def test_frozen_model_policy_carries_reach_2048_task():
    policy = FrozenModelPolicy(
        TinyModel(),
        TinyTokenizer(),
        dict(
            policy_id="test",
            utility="threshold",
            threshold=2048,
            task="reach_2048",
            continuation_policy_id="heuristic:test",
        ),
        configuration(),
    )
    assert policy.controller.task == "reach_2048"
    assert policy.controller.continuation_policy_id == "heuristic:test"
    default = FrozenModelPolicy(
        TinyModel(),
        TinyTokenizer(),
        dict(policy_id="test", utility="log_tile", threshold=2048),
        configuration(),
    )
    assert default.controller.task == "terminal_max_tile"
    assert default.controller.continuation_policy_id is None


def tiny_game(seed):
    return Game(
        seed,
        board=((2, 2, 8, 16), (8, 16, 32, 64), (16, 32, 64, 128), (32, 64, 128, 256)),
    )


def configuration():
    return load_config("configs/model.yaml") | load_config("configs/online_smoke.yaml")


def test_async_collector_matches_synchronous_order_and_content():
    config = configuration()
    policy = FrozenHeuristic(config["heuristic_config"])
    expected_collector = OnlineCollector(policy, config, 0)
    expected = expected_collector.collect(2)
    collector = AsyncOnlineCollector(policy, config, 0, 2, 1)
    try:
        actual = collector.collect(2)
        assert actual == expected
        assert collector.counters == expected_collector.counters
    finally:
        collector.close()


class TinyTokenizer:
    def __call__(self, paths, **kwargs):
        ids = torch.tensor(
            [[DESCRIPTIONS.index(path.split("[CANDIDATE]\n")[1]) + 1] for path in paths]
        )
        return dict(input_ids=ids, attention_mask=torch.ones_like(ids))


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Embedding(8, 4)
        self.head = nn.Linear(4, 1)

    def forward(self, batch):
        values = self.head(self.backbone(batch["tokens"]["input_ids"][:, 0])).squeeze(
            -1
        )
        logits = nn.utils.rnn.pad_sequence(
            [values[a:b] for a, b in zip(batch["offsets"][:-1], batch["offsets"][1:])],
            batch_first=True,
        )
        return masked_log_probs(logits, batch["mask"])


def test_online_seed_streams_and_events():
    c = configuration()
    policy = FrozenHeuristic()
    a = OnlineCollector(policy, c, 0, game_factory=tiny_game)
    b = OnlineCollector(policy, c, 0, game_factory=tiny_game)
    rows = a.collect(4)
    torch.rand(20)  # Predictive draws must not change simulator randomness.
    assert rows == b.collect(4)
    assert all("q" not in row and row["rollout_steps"] >= 1 for row in rows)
    assert len({row["rollout_seed"] for row in rows}) == len(rows)
    assert (
        len(
            {
                stream_seed(17, 0, split)
                for split in ("train", "holdout", "mc", "promotion", "test")
            }
        )
        == 5
    )
    assert rows != OnlineCollector(policy, c, 1, game_factory=tiny_game).collect(4)
    validate_events(rows, policy.policy_id, 0)
    for changed in (
        dict(policy_id="stale"),
        dict(generation=1),
        dict(stream="mc"),
        dict(q=[1.0, 0.0]),
    ):
        with pytest.raises(ValueError):
            validate_events([rows[0] | changed], policy.policy_id, 0)


def test_counterfactual_actions_and_mc():
    c = configuration()
    policy = FrozenHeuristic()
    collector = OnlineCollector(policy, c, 0, game_factory=tiny_game)
    questions = collector.collect_questions(1)
    assert [r["action"] for r in questions] == tiny_game(0).legal_actions
    assert all(r["board"] == questions[0]["board"] for r in questions)
    assert all(r["steps"] == 0 for r in questions)
    before = copy.deepcopy(questions)
    events = terminal_feedback(questions, policy)
    assert questions == before
    assert all(
        event["terminal_max_tile"] >= max(map(max, event["board"])) for event in events
    )
    mc, steps = mc_reference(questions, policy, 4, 71, 2)
    again, _ = mc_reference(questions, policy, 4, 71, 3)
    assert mc == again and steps >= len(questions) * 4
    assert all(sum(r["counts"]) == 4 and sum(r["q"]) == 1 for r in mc)
    with pytest.raises(ValueError):
        terminal_feedback([questions[0] | dict(policy_id="bad")], policy)
    with pytest.raises(ValueError):
        mc_reference(events, policy, 4, 71)
    retargeted = retarget_questions(events, policy.policy_id, 1, 17, "validation")
    assert all("observed_outcome" not in r and r["generation"] == 1 for r in retargeted)


def test_feedback_calls_the_frozen_continuation():
    class RecordingPolicy:
        policy_id = "recording"

        def __init__(self):
            self.calls = 0
            self.delegate = FrozenHeuristic()

        def choose_many(self, games):
            self.calls += 1
            return self.delegate.choose_many(games)

    policy = RecordingPolicy()
    row = dict(
        board=((2, 2, 4, 0), (8, 16, 32, 64), (16, 32, 64, 128), (32, 64, 128, 256)),
        score=0,
        steps=0,
        action="LEFT",
        rollout_seed=17,
        policy_id=policy.policy_id,
    )
    event = terminal_feedback([row], policy)[0]
    assert policy.calls > 0 and event["rollout_steps"] > 1


def test_snapshot_is_independent_of_learner():
    model = TinyModel()
    tokenizer = TinyTokenizer()
    c = configuration()
    frozen = FrozenModelPolicy(
        copy.deepcopy(model),
        tokenizer,
        dict(policy_id="test", utility="log_tile", threshold=2048),
        c,
    )
    original = {k: v.clone() for k, v in frozen.model.state_dict().items()}
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(10)
    assert all(not p.requires_grad for p in frozen.model.parameters())
    assert not frozen.model.training
    assert all(
        torch.equal(v, original[k]) for k, v in frozen.model.state_dict().items()
    )


def test_batched_controller_question_mapping(monkeypatch):
    def fake_predict(model, tokenizer, rows, microbatch, max_length, precision="fp32"):
        p = np.zeros((len(rows), 7))
        for i, row in enumerate(rows):
            p[i, 4] = 0.9 if row["action"] == "RIGHT" else 0.1
            p[i, 0] = 1 - p[i, 4]
        return p

    monkeypatch.setattr(controller_module, "predict", fake_predict)
    controller = JevController(None, None)
    games = [tiny_game(1), Game(2)]
    assert controller.choose_many(games) == ["RIGHT", "RIGHT"]


def test_promotion_and_action_regret():
    old = dict(
        mean_log_tile=8.0,
        records=[
            dict(seed=1, score=100, max_tile=256),
            dict(seed=2, score=100, max_tile=256),
        ],
    )
    new = dict(
        mean_log_tile=8.5,
        records=[
            dict(seed=1, score=110, max_tile=512),
            dict(seed=2, score=100, max_tile=256),
        ],
    )
    decision = promotion_decision(old, new, 0.25)
    assert decision["accepted"] and decision["metric"] == "mean_log_tile"
    assert not promotion_decision(old, old, 0)["accepted"]
    assert not promotion_decision(old, new, 0.5)["accepted"]
    with pytest.raises(ValueError):
        promotion_decision(
            old,
            dict(new, records=[dict(seed=3, score=105, max_tile=512)]),
            0,
        )
    p = np.zeros((2, 7))
    q = p.copy()
    p[0, 4] = 1
    p[1, 0] = 1
    q[0, 0] = 1
    q[1, 4] = 1
    result = action_ranking_metrics(
        p, [dict(state_id="s", q=row.tolist()) for row in q], "threshold", 2048
    )
    assert result["mc_greedy_agreement"] == 0 and result["mc_utility_regret"] == 1


def test_online_config_requires_no_offline_data():
    c = configuration()
    validate_online_config(c)
    for update in (
        dict(data_dir="data/main"),
        dict(microbatch=3),
        dict(mc_rollouts=1),
        dict(behavior_epsilon=1.1),
        dict(min_optimizer_steps_per_policy=30),
        dict(utility="threshold"),
    ):
        with pytest.raises(ValueError):
            validate_online_config(c | update)
    c = load_config("configs/model.yaml") | load_config("configs/online.yaml")
    c["objective"] = "paired_pg"
    validate_online_config(c)
    assert c["mode"] == "online"


def test_two_generation_online_pipeline(tmp_path, monkeypatch):
    torch.manual_seed(17)
    torch.set_num_threads(1)
    decisions = iter([False, True, False, False])

    def exercise_both_paths(incumbent, candidate, margin):
        result = promotion_decision(incumbent, candidate, margin)
        result["accepted"] = next(decisions)
        return result

    monkeypatch.setattr(online_trainer, "promotion_decision", exercise_both_paths)
    monkeypatch.setattr(online_trainer, "plot_reliability", lambda *args: None)
    c = configuration() | dict(
        max_policy_generations=2,
        min_optimizer_steps_per_policy=1,
        max_optimizer_steps_per_policy=2,
        promotion_interval_steps=1,
        max_total_optimizer_steps=4,
    )
    summaries = run_online(
        c,
        tmp_path,
        model=TinyModel(),
        tokenizer=TinyTokenizer(),
        game_factory=tiny_game,
    )
    assert len(summaries) == 2
    assert summaries[0]["accepted"] and not summaries[1]["accepted"]
    assert summaries[0]["status"] == "promoted"
    assert summaries[1]["status"] == "policy_stalled"
    assert summaries[1]["target_policy_id"] == summaries[0]["active_policy_id"]
    assert summaries[1]["active_policy_id"] == summaries[0]["active_policy_id"]
    first = [
        json.loads(line)
        for line in (tmp_path / "generation_000/events.jsonl").read_text().splitlines()
    ]
    second = [
        json.loads(line)
        for line in (tmp_path / "generation_001/events.jsonl").read_text().splitlines()
    ]
    assert not {r["source_game"] for r in first} & {r["source_game"] for r in second}
    assert first[0]["policy_id"] != second[0]["policy_id"]
    assert len(first) == len(second) == 2 * c["effective_batch"]
    assert summaries[0]["optimizer_steps"] == summaries[1]["optimizer_steps"] == 2
    assert audit(tmp_path)["success"]
    records = [
        json.loads(line)
        for line in (tmp_path / "training.jsonl").read_text().splitlines()
    ]
    assert all("peak_gpu_gib" not in row for row in records)
    excluded_fields = {
        "step_seconds", "questions_per_second", "target_policy_id", "elapsed_seconds"
    }
    assert all(excluded_fields.isdisjoint(row) for row in records)
    assert not (tmp_path / "generation_000/training_stats.json").exists()
    assert all(
        "collection_seconds" in row and row["environment"]["terminal_rollouts"] > 0
        for row in records
    )
    assert (tmp_path / "generation_001/controller_test.json").exists()
    assert (tmp_path / "generation_000/initial_validation_predictions.json").exists()
    assert (tmp_path / "generation_000/final_validation_predictions.json").exists()
    with pytest.raises(FileExistsError):
        run_online(c, tmp_path, model=TinyModel(), tokenizer=TinyTokenizer())
