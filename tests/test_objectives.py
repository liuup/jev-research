import itertools
from fractions import Fraction

import pytest
import torch

from jevmath.env import State
from jevmath.objectives import loss, masked_log_probs, paired_pg
from jevmath.serialization import candidates, question_row, serialize

ROW = dict(
    id="demo",
    question="Ann has 3 apples and buys 4 more.",
    numbers=((Fraction(3), "q1"), (Fraction(4), "q2")),
    gold=Fraction(7),
)


def test_masking_and_direct_losses():
    logits = torch.tensor([[1.0, 2.0, 9.0]])
    mask = torch.tensor([[True, True, False]])
    logp = masked_log_probs(logits, mask)
    assert logp.exp()[0, 2] == 0
    torch.testing.assert_close(logp.exp().sum(-1), torch.ones(1))
    torch.testing.assert_close(logp.exp()[:, :2], logits[:, :2].softmax(-1))
    targets = torch.tensor([1])
    torch.testing.assert_close(loss(logp, targets, "ce"), -logp[0, 1])
    torch.testing.assert_close(
        loss(logp, targets, "brier"),
        ((logp.exp() - torch.tensor([[0.0, 1.0, 0.0]])) ** 2).sum(),
    )
    with pytest.raises(ValueError):
        masked_log_probs(logits, torch.zeros_like(mask))
    with pytest.raises(ValueError, match="observed outcome"):
        loss(logp, torch.tensor([-1]), "ce")


@pytest.mark.parametrize("k,m", [(2, 2), (2, 3), (3, 2)])
@pytest.mark.parametrize("baseline", ["conditional", "zero"])
def test_paired_pg_gradient_identity(k, m, baseline):
    """The estimator's exact gradient is the gradient of the squared L2 objective."""
    logits = torch.linspace(-0.6, 0.8, k, dtype=torch.float64, requires_grad=True)
    p = logits.softmax(0)
    q = torch.arange(1, k + 1, dtype=torch.float64)
    q = q / q.sum()
    expected = torch.zeros_like(logits)
    for draws in itertools.product(range(k), repeat=m):
        draw_tensor = torch.tensor([draws])
        weight = p[list(draws)].prod().detach()
        for target in range(k):
            value = paired_pg(
                logits.log_softmax(0)[None],
                torch.tensor([target]),
                baseline=baseline,
                draws=draw_tensor,
            )
            expected += weight * q[target] * torch.autograd.grad(value, logits)[0]
    actual = torch.autograd.grad(((p - q) ** 2).sum(), logits)[0]
    torch.testing.assert_close(expected, actual, atol=1e-14, rtol=1e-11)


def test_paired_pg_is_permutation_equivariant():
    logits = torch.tensor([[0.2, -0.3, 0.7]], dtype=torch.float64, requires_grad=True)
    draws = torch.tensor([[0, 2, 2, 1]])
    targets = torch.tensor([2])
    permutation = torch.tensor([2, 0, 1])
    inverse = permutation.argsort()
    plain = paired_pg(logits.log_softmax(-1), targets, draws=draws)
    permuted = paired_pg(
        logits[:, permutation].log_softmax(-1), inverse[targets], draws=inverse[draws]
    )
    torch.testing.assert_close(plain, permuted)
    torch.testing.assert_close(
        torch.autograd.grad(plain, logits)[0],
        torch.autograd.grad(permuted, logits)[0],
    )


def test_paired_pg_diagnostics_do_not_change_the_gradient():
    logits = torch.tensor([[0.2, -0.3]], dtype=torch.float64, requires_grad=True)
    draws = torch.tensor([[0, 1, 1, 0]])
    diagnostics = {}
    plain = paired_pg(logits.log_softmax(-1), torch.tensor([1]), draws=draws)
    logged = paired_pg(
        logits.log_softmax(-1), torch.tensor([1]), draws=draws, diagnostics=diagnostics
    )
    torch.testing.assert_close(plain, logged)
    torch.testing.assert_close(
        torch.autograd.grad(plain, logits)[0],
        torch.autograd.grad(logged, logits)[0],
    )
    assert all(not value.requires_grad for value in diagnostics.values())
    assert diagnostics["sampled_hit_rate"].item() == 0.5


def test_serialization_contains_state_action_and_question():
    state = State.initial(ROW)
    candidate = state.candidates()[0]
    row = question_row(state, candidate, "policy-17")
    assert candidates(row) == ["yes", "no"]
    prompt = serialize(row | dict(candidate_id="yes"))
    assert ROW["question"] in prompt
    assert state.describe(candidate) in prompt
    assert "policy-17" in prompt
    assert "FrozenContinuationPolicy" in prompt
    assert prompt.rstrip().endswith("YES")
    assert "no" in candidates(row)
    with pytest.raises(ValueError):
        serialize(row | dict(candidate_id="maybe"))
