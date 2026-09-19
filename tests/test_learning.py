"""Prompts, candidate paths, batching and the objectives behind the training step."""

from fractions import Fraction

import pytest
import torch

from jevmath.dataset import collate
from jevmath.env import State
from jevmath.objectives import loss, masked_log_probs, paired_pg
from jevmath.serialization import candidates, question_row, serialize
from jevmath.tiny import tiny_model_and_tokenizer


def puzzle(numbers):
    return dict(puzzle_id="_".join(map(str, numbers)), numbers=list(numbers), target=24)


def first_row(numbers=(3, 3, 8, 8), key="/:8:3"):
    state = State.initial(puzzle(list(numbers)))
    action = next(a for a in state.actions() if a["key"] == key)
    return state, action, question_row(state, action, "24game:seed17:g0")


def test_prompt_states_target_remaining_action_and_policy():
    state, action, row = first_row()
    text = serialize(row | dict(candidate_id="yes"))
    assert "[STATE]" in text and "Target: 24" in text
    assert "Remaining: 3 3 8 8" in text
    assert "Steps already taken:\n(none)" in text
    assert "[CONTINUATION]\nUse the frozen continuation policy." in text
    assert "24game:seed17:g0" not in text
    assert "[ACTION]\n8 / 3 = 8/3" in text
    assert "will the final value equal 24" in text
    assert "[OPTIONS]\nYES\nNO" in text
    assert text.rstrip().endswith("[CANDIDATE]\nYES")
    assert serialize(row | dict(candidate_id="no")).rstrip().endswith("[CANDIDATE]\nNO")


def test_prompt_carries_previous_steps_and_the_remaining_operation_count():
    state = State.initial(puzzle([3, 3, 8, 8]))
    state = state.apply(next(a for a in state.actions() if a["key"] == "/:8:3"))
    action = next(a for a in state.actions() if a["key"] == "-:3:8/3")
    row = question_row(state, action, "24game:seed17:g0")
    text = serialize(row | dict(candidate_id="yes"))
    assert "Steps already taken:\n1. 8 / 3 = 8/3" in text
    assert "Remaining: 8/3 3 8" in text
    assert "after 2 more operations?" in text
    deep = state.apply(action)
    action3 = deep.actions()[0]
    assert "after 1 more operation?" in serialize(
        question_row(deep, action3, "p") | dict(candidate_id="no")
    )


def test_unknown_candidate_is_rejected():
    _, _, row = first_row()
    with pytest.raises(ValueError):
        serialize(row | dict(candidate_id="maybe"))
    with pytest.raises(ValueError):
        candidates(dict(candidate_ids=["yes", "yes"]))


def test_collate_builds_one_group_per_question_with_two_paths():
    _, _, row = first_row()
    tokenizer = tiny_model_and_tokenizer()[1]
    batch = collate([row], tokenizer, 512)
    assert batch["tokens"]["input_ids"].shape[0] == 2
    assert batch["offsets"] == [0, 2]
    assert batch["mask"].tolist() == [[True, True]]
    assert batch["state_ids"] == [row["state_id"]]
    assert batch["action_keys"] == ["/:8:3"]
    assert batch["targets"].tolist() == [-1]  # no observed outcome yet


def test_collate_targets_point_at_the_observed_outcome():
    _, _, row = first_row()
    tokenizer = tiny_model_and_tokenizer()[1]
    yes = collate([row | dict(observed_outcome="yes")], tokenizer, 512)
    no = collate([row | dict(observed_outcome="no")], tokenizer, 512)
    assert yes["targets"].tolist() == [0]
    assert no["targets"].tolist() == [1]


def test_collate_refuses_rows_that_exceed_max_length():
    _, _, row = first_row()
    tokenizer = tiny_model_and_tokenizer()[1]
    with pytest.raises(ValueError, match="max_length"):
        collate([row], tokenizer, 4)


def test_masked_log_probs_normalizes_over_paths():
    logits = torch.tensor([[1.0, 3.0], [2.0, -5.0]])
    mask = torch.tensor([[True, True], [True, False]])
    logp = masked_log_probs(logits, mask)
    assert torch.allclose(logp.exp().sum(-1)[0], torch.tensor(1.0))
    assert logp[1, 1].item() == -torch.inf
    with pytest.raises(ValueError):
        masked_log_probs(logits, torch.zeros_like(mask))


def test_objectives_need_an_observed_outcome():
    _, _, row = first_row()
    tokenizer = tiny_model_and_tokenizer()[1]
    model, _ = tiny_model_and_tokenizer()
    logp = model(collate([row], tokenizer, 512))
    for objective in ("ce", "brier", "paired_pg"):
        with pytest.raises(ValueError, match="observed outcome"):
            loss(logp, torch.tensor([-1]), objective, 8)


def test_brier_and_ce_match_their_definitions():
    logits = torch.tensor([[0.3, -0.7], [-1.0, 2.0]], dtype=torch.float64)
    logp = torch.log_softmax(logits, -1)
    targets = torch.tensor([0, 1])
    probabilities = logp.exp()
    one_hot = torch.nn.functional.one_hot(targets, 2).double()
    assert torch.allclose(loss(logp, targets, "brier"), ((probabilities - one_hot) ** 2).sum(-1).mean())
    assert torch.allclose(loss(logp, targets, "ce"), -logp.gather(1, targets[:, None]).mean())


def test_paired_pg_estimator_is_unbiased_for_the_brier_gradient():
    """The estimator averages to the exact squared-error gradient over many draws."""
    torch.manual_seed(0)
    logits = torch.tensor([[0.4, -0.2]], dtype=torch.float64, requires_grad=True)
    target = torch.tensor([0])
    logp = torch.log_softmax(logits, -1)
    exact = torch.autograd.grad(loss(logp, target, "brier"), logits)[0]
    total = torch.zeros_like(exact)
    draws = 4000
    for _ in range(draws):
        graph_logits = torch.tensor(logits.detach().tolist(), dtype=torch.float64, requires_grad=True)
        graph_logp = torch.log_softmax(graph_logits, -1)
        gradient = torch.autograd.grad(paired_pg(graph_logp, target, 8), graph_logits)[0]
        total += gradient
    average = total / draws
    assert torch.allclose(average, exact, atol=2e-2), (average, exact)


def test_paired_pg_advantage_vanishes_for_a_deterministic_policy():
    """Same-class draws cancel exactly: that is the saturation failure mode to watch."""
    logits = torch.tensor([[20.0, -20.0]], dtype=torch.float64, requires_grad=True)
    logp = torch.log_softmax(logits, -1)
    diagnostics = {}
    gradient = torch.autograd.grad(
        paired_pg(logp, torch.tensor([1]), 32, diagnostics=diagnostics), logits
    )[0]
    assert torch.allclose(gradient, torch.zeros_like(gradient), atol=1e-6)
    assert diagnostics["sampled_collision_rate"].item() > 0.99
