from fractions import Fraction

import numpy as np
import pytest
import torch
from torch import nn

from jevmath import trainer as trainer_module
from jevmath.config import load_config
from jevmath.env import State
from jevmath.objectives import masked_log_probs
from jevmath.rollout import (
    action_distribution,
    collect,
    record_json,
    rollout_batch,
    score_candidates,
)
from jevmath.trainer import run, validate_config

PROBLEMS = [
    dict(
        id=f"p{index}",
        question=f"Problem {index}: start from {index + 2} and add {index + 3}.",
        numbers=((Fraction(index + 2), "q1"), (Fraction(index + 3), "q2")),
        gold=Fraction(2 * index + 5),
    )
    for index in range(4)
]


class TinyTokenizer:
    def __call__(self, paths, padding=True, truncation=False, return_tensors=None):
        ids = torch.tensor([[1 + len(path) % 251] for path in paths])
        return dict(input_ids=ids, attention_mask=torch.ones_like(ids))


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Embedding(512, 8)
        self.head = nn.Linear(8, 1)

    def forward(self, batch):
        values = self.head(self.backbone(batch["tokens"]["input_ids"][:, 0])).squeeze(-1)
        logits = torch.nn.utils.rnn.pad_sequence(
            [values[a:b] for a, b in zip(batch["offsets"][:-1], batch["offsets"][1:])],
            batch_first=True,
        )
        return masked_log_probs(logits, batch["mask"])


def tiny_config(**overrides):
    config = load_config("configs/model.yaml") | load_config("configs/smoke.yaml")
    return config | dict(
        generations=1,
        problems_per_generation=2,
        rollouts_per_problem=2,
        max_steps=2,
        steps_per_generation=1,
        effective_batch=2,
        microbatch=1,
        warmup_steps=1,
        validation_problems=2,
        inference_questions=2,
        action_reference_problems=0,
        validation_fraction=0.25,
    ) | overrides


def test_action_distribution_normalises_and_respects_temperature():
    state = State.initial(PROBLEMS[0])
    candidates = state.candidates()
    probabilities = {candidate["key"]: 0.25 for candidate in candidates}
    cold = action_distribution(probabilities, candidates, 1.0)
    assert sum(cold.values()) == pytest.approx(1.0)
    assert all(value == pytest.approx(1 / len(candidates)) for value in cold.values())
    skewed = {key: (0.9 if index == 0 else 0.1) for index, key in enumerate(probabilities)}
    hot = action_distribution(skewed, candidates, 0.25)
    assert hot[candidates[0]["key"]] > cold[candidates[0]["key"]]
    with pytest.raises(ValueError):
        action_distribution(probabilities, candidates, 0.0)


def test_score_candidates_covers_every_action():
    state = State.initial(PROBLEMS[0])
    candidates = state.candidates()
    config = tiny_config(policy_id="test")
    probabilities = score_candidates(
        TinyModel(), TinyTokenizer(), [(state, candidates)], config
    )[0]
    assert set(probabilities) == {candidate["key"] for candidate in candidates}
    assert all(0.0 <= value <= 1.0 for value in probabilities.values())


def test_rollout_is_deterministic_and_reports_outcomes():
    config = tiny_config(policy_id="test")
    model, tokenizer, seeds = TinyModel(), TinyTokenizer(), [11, 12]
    first = rollout_batch(model, tokenizer, PROBLEMS[:2], config, seeds)
    second = rollout_batch(model, tokenizer, PROBLEMS[:2], config, seeds)
    assert [record_json(r) for r in first] == [record_json(r) for r in second]
    for record in first:
        assert record["answer"] is not None and isinstance(record["correct"], bool)
        assert len(record["steps"]) <= config["max_steps"] + 1
        for step in record["steps"]:
            assert sum(step["distribution"].values()) == pytest.approx(1.0)
            assert step["chosen"] in step["distribution"]
        assert record["gold"] == str(next(p for p in PROBLEMS if p["id"] == record["problem_id"])["gold"])


def test_collect_groups_rollouts_and_labels_rows():
    config = tiny_config(policy_id="test", rollouts_per_problem=2)
    problems = PROBLEMS[:2]
    seeds = {
        (problem["id"], repeat): 100 + index
        for index, problem in enumerate(problems)
        for repeat in range(2)
    }
    records, rows = collect(TinyModel(), TinyTokenizer(), problems, config, seeds)
    assert len(records) == 4
    assert len(rows) == sum(len(record["steps"]) for record in records)
    by_problem = {}
    for row in rows:
        assert row["observed_outcome"] in ("yes", "no")
        assert row["state_id"].startswith(row["row_id"])
        by_problem.setdefault(row["row_id"], set()).add(row["observed_outcome"])
    for outcomes in by_problem.values():
        assert outcomes == {"no"} or outcomes == {"yes"}


def test_trainer_generation_loop_writes_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(trainer_module, "plot_reliability", lambda *args: None)
    config = tiny_config()
    validate_config(config)
    summaries = run(
        config,
        tmp_path / "run",
        model=TinyModel(),
        tokenizer=TinyTokenizer(),
        rows=PROBLEMS,
    )
    assert len(summaries) == 1
    summary = summaries[0]
    assert summary["rollout"]["rollouts"] == 4
    assert summary["rollout"]["training_rows"] > 0
    assert 0.0 <= summary["validation"]["final_answer_accuracy"] <= 1.0
    assert summary["validation"]["questions"] > 0
    assert "curve" in summary["validation"]
    for name in (
        "resolved_config.json",
        "resolved_config.yaml",
        "generations.json",
        "summary.json",
        "training.jsonl",
        "checkpoint.pt",
    ):
        assert (tmp_path / "run" / name).exists(), name
    for name in (
        "trajectories.jsonl",
        "rows.jsonl",
        "rollout_metrics.json",
        "validation_metrics.json",
        "validation_rows.jsonl",
    ):
        assert (tmp_path / "run" / "generation_000" / name).exists(), name
    with pytest.raises(FileExistsError):
        run(config, tmp_path / "run", model=TinyModel(), tokenizer=TinyTokenizer(), rows=PROBLEMS)


def test_objective_variants_run_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(trainer_module, "plot_reliability", lambda *args: None)
    for objective in ("ce", "brier"):
        summaries = run(
            tiny_config(objective=objective),
            tmp_path / objective,
            model=TinyModel(),
            tokenizer=TinyTokenizer(),
            rows=PROBLEMS,
        )
        assert summaries[0]["mean_loss"] is not None
        assert np.isfinite(summaries[0]["mean_loss"])
