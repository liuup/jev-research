"""Metrics, oracle separation and the full generation loop on CPU with the tiny model."""

import json
import random
from pathlib import Path

import numpy as np
import pytest

from jevmath.config import load_config
from jevmath.evaluation import (
    observed_metrics,
    oracle_separation,
    reliability,
    risk_coverage,
)
from jevmath.puzzles import load_oracle, load_puzzles
from jevmath.tiny import tiny_model_and_tokenizer
from jevmath.trainer import (
    evaluate_policy,
    generation_config,
    generation_policy_id,
    learning_rate_scale,
    run,
    sample_puzzles,
    validate_config,
)

DATA = Path("data/24game")


def rows_for(labels):
    return [
        dict(candidate_ids=["yes", "no"], observed_outcome=label, action_key=f"+:1:{index}", depth=0)
        for index, label in enumerate(labels)
    ]


def test_binary_brier_and_nll_follow_their_definitions():
    rows = rows_for(["yes", "no", "no", "no"])
    predictions = np.array([[0.75, 0.25], [0.5, 0.5], [0.25, 0.75], [0.1, 0.9]])
    metrics = observed_metrics(predictions, rows)
    yes = predictions[:, 0]
    event = np.array([1.0, 0.0, 0.0, 0.0])
    assert metrics["binary_brier"] == pytest.approx(float(((yes - event) ** 2).mean()))
    assert metrics["observed_brier"] == pytest.approx(
        float(((predictions - np.eye(2)[(1 - event).astype(int)]) ** 2).sum(-1).mean())
    )
    assert metrics["yes_rate"] == 0.25
    assert metrics["predicted_success_probability"] == pytest.approx(float(yes.mean()))
    assert metrics["observed_nll"] == pytest.approx(
        float(
            -(
                event * np.log(yes)
                + (1 - event) * np.log(1 - yes)
            ).mean()
        )
    )


def test_ece_is_zero_for_a_perfectly_calibrated_split():
    probability = np.array([0.0, 0.2, 0.4, 1.0])
    event = np.array([0.0, 0.2, 0.4, 1.0])
    ece, diagram = reliability(probability, event)
    assert ece == pytest.approx(0.0, abs=1e-9)
    assert sum(bucket["count"] for bucket in diagram) == len(probability)
    ece, _ = reliability(np.array([0.9, 0.9, 0.9, 0.9]), np.array([0.0, 0.0, 0.0, 0.0]))
    assert ece == pytest.approx(0.9)


def test_risk_coverage_rewards_confident_correct_rows():
    rows = rows_for(["yes", "yes", "no", "no"])
    good = np.array([[0.9, 0.1], [0.8, 0.2], [0.1, 0.9], [0.2, 0.8]])
    bad = np.array([[0.1, 0.9], [0.2, 0.8], [0.9, 0.1], [0.8, 0.2]])
    assert risk_coverage(good, rows)["aurc"] < risk_coverage(bad, rows)["aurc"]


def test_oracle_separation_uses_the_stored_first_step_actions():
    oracle = {("3_3_8_8", 0): {"/:8:3": True, "+:3:3": False}}
    rows = [
        dict(candidate_ids=["yes", "no"], observed_outcome="yes", puzzle_id="3_3_8_8", depth=0, action_key="/:8:3"),
        dict(candidate_ids=["yes", "no"], observed_outcome="no", puzzle_id="3_3_8_8", depth=0, action_key="+:3:3"),
        dict(candidate_ids=["yes", "no"], observed_outcome="no", puzzle_id="3_3_8_8", depth=1, action_key="+:3:3"),
    ]
    predictions = np.array([[0.8, 0.2], [0.2, 0.8], [0.5, 0.5]])
    metrics = oracle_separation(predictions, rows, oracle)
    assert metrics == dict(
        states_with_oracle=2,
        solving_actions=1,
        dead_actions=1,
        mean_p_yes_solving=0.8,
        mean_p_yes_dead=0.2,
        separation=pytest.approx(0.6),
    )


def smoke_config(puzzles_per_generation=8, steps=4, generations=1):
    config = load_config("configs/model.yaml") | load_config("configs/smoke.yaml")
    return dict(
        config,
        data_dir=str(DATA),
        puzzles_per_generation=puzzles_per_generation,
        steps_per_generation=steps,
        generations=generations,
        validation_puzzles=4,
    )


def test_generation_labels_and_schedule():
    config = smoke_config()
    assert generation_policy_id(config, 0) == "24game:seed17:solver0.5:g0"
    assert generation_policy_id(config, 3) == "24game:seed17:g3"
    assert generation_config(config, 0)["solver_mix"] == 0.5
    assert generation_config(config, 1)["solver_mix"] == 0.0
    assert learning_rate_scale(0, 100, 10) == 0.1
    assert learning_rate_scale(10, 100, 10) == 1.0
    assert learning_rate_scale(100, 100, 10) == 0.0


def test_config_validation_rejects_bad_values():
    config = smoke_config()
    validate_config(config)
    for key, value in (
        ("microbatch", 0),
        ("objective", "nope"),
        ("baseline", "nope"),
        ("inference_precision", "int8"),
        ("rollout_mode", "noisy"),
        ("init_policy", "solver"),
        ("temperature", 0.0),
    ):
        broken = dict(config, **{key: value})
        with pytest.raises(ValueError):
            validate_config(broken)
    with pytest.raises(ValueError):
        validate_config(dict(config, microbatch=config["effective_batch"] + 1))


def test_sample_puzzles_is_deterministic_and_handles_all_train_puzzles():
    train = load_puzzles(DATA / "train.jsonl")
    first = sample_puzzles(train, 8, random.Random(1))
    assert first == sample_puzzles(train, 8, random.Random(1))
    assert len(sample_puzzles(train, len(train), random.Random(1))) == len(train)
    assert len(sample_puzzles(train, len(train) + 5, random.Random(1))) == len(train) + 5


@pytest.mark.skipif(not (DATA / "train.jsonl").exists(), reason="dataset not built")
def test_generation_runs_end_to_end_on_cpu(tmp_path):
    config = smoke_config()
    model, tokenizer = tiny_model_and_tokenizer()
    summaries = run(config, tmp_path / "run", model=model, tokenizer=tokenizer)
    assert len(summaries) == 1
    summary = summaries[0]
    rollout = summary["rollout"]
    dev = summary["dev_calibration"]
    controller = summary["dev_controller"]
    assert rollout["puzzles"] == 8 and rollout["rollouts"] == 16
    assert rollout["training_rows"] > 0
    assert 0.0 <= rollout["row_yes_rate"] <= 1.0
    assert summary["frozen_dev_rollout"]["puzzles"] == 4
    assert summary["frozen_dev_rollout"]["rollouts"] == 8
    assert 0.0 <= summary["frozen_dev_rollout"]["solved_rate"] <= 1.0
    assert dev["questions"] > 0
    assert 0.0 <= dev["binary_brier"] <= 2.0
    assert dev["oracle_separation"]["separation"] is None or -1 <= dev["oracle_separation"]["separation"] <= 1
    assert 0.0 <= controller["solved_rate"] <= 1.0
    steps = [
        json.loads(line)
        for line in (tmp_path / "run" / "training.jsonl").read_text().splitlines()
    ]
    assert len(steps) == config["steps_per_generation"]
    assert all(np.isfinite(step["loss"]) for step in steps)
    assert (tmp_path / "run" / "checkpoint.pt").exists()
    assert (tmp_path / "run" / "summary.json").exists()
    saved = json.loads((tmp_path / "run" / "resolved_config.json").read_text())
    assert saved["train_puzzles"] == len(load_puzzles(DATA / "train.jsonl")) == 954
    assert saved["dev_puzzles"] == config["validation_puzzles"]
    with pytest.raises(FileExistsError):
        run(config, tmp_path / "run", model=model, tokenizer=tokenizer)


@pytest.mark.skipif(not (DATA / "dev.jsonl").exists(), reason="dataset not built")
def test_evaluate_policy_reports_calibration_and_oracle_metrics():
    model, tokenizer = tiny_model_and_tokenizer()
    config = smoke_config()
    oracle = load_oracle(DATA)
    records, rows, metrics = evaluate_policy(
        model,
        tokenizer,
        load_puzzles(DATA / "dev.jsonl")[:4],
        generation_config(config, 0),
        mode="greedy",
        oracle=oracle,
    )
    assert len(records) == 4
    assert all(len(record["steps"]) == 3 for record in records)
    assert metrics["questions"] == len(rows) > 0
    assert metrics["first_action_solvable_rate"] is not None
    assert metrics["oracle_separation"]["states_with_oracle"] == 4
