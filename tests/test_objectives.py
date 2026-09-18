import itertools

import pytest
import torch

from jev2048.objectives import loss, masked_log_probs, paired_pg


def test_pg_diagnostics_preserve_gradient_and_reward():
    z = torch.tensor([[0.2, -0.3, 0.7]], dtype=torch.float64, requires_grad=True)
    draws = torch.tensor([[0, 2, 2, 1]])
    y = torch.tensor([2])
    diagnostics = {}
    plain = paired_pg(z.log_softmax(-1), y, draws=draws)
    logged = paired_pg(z.log_softmax(-1), y, draws=draws, diagnostics=diagnostics)
    torch.testing.assert_close(plain, logged)
    torch.testing.assert_close(
        torch.autograd.grad(plain, z)[0], torch.autograd.grad(logged, z)[0]
    )
    assert all(not value.requires_grad for value in diagnostics.values())
    assert diagnostics["sampled_hit_rate"].item() == 0.5
    assert diagnostics["sampled_collision_rate"].item() == pytest.approx(1 / 6)
    assert diagnostics["sampled_reward"].item() == pytest.approx(5 / 6)


@pytest.mark.parametrize("k,m", [(2, 2), (3, 2), (2, 3)])
@pytest.mark.parametrize("baseline", ["conditional", "zero"])
def test_exact_gradient(k, m, baseline):
    z = torch.linspace(-0.6, 0.8, k, dtype=torch.float64, requires_grad=True)
    p = z.softmax(0)
    q = torch.arange(1, k + 1, dtype=torch.float64)
    q = q / q.sum()
    expected = torch.zeros_like(z)
    reward = 0.0
    for draws in itertools.product(range(k), repeat=m):
        a = torch.tensor([draws])
        weight = p[list(draws)].prod().detach()
        counts = torch.bincount(a[0], minlength=k)
        for y in range(k):
            l = paired_pg(
                z.log_softmax(0)[None], torch.tensor([y]), baseline=baseline, draws=a
            )
            g = torch.autograd.grad(l, z, retain_graph=True)[0]
            expected += weight * q[y] * g
            r = 2 / m * sum(x == y for x in draws) - (counts * (counts - 1)).sum() / (
                m * (m - 1)
            )
            reward += weight * q[y] * r
    actual = torch.autograd.grad(((p - q) ** 2).sum(), z)[0]
    torch.testing.assert_close(expected, actual, atol=1e-14, rtol=1e-13)
    torch.testing.assert_close(
        reward, 2 * (p * q).sum() - (p * p).sum(), atol=1e-7, rtol=1e-7
    )


def test_scores_mask_permutation():
    z = torch.tensor([[1.0, 2.0, 9.0]])
    mask = torch.tensor([[True, True, False]])
    lp = masked_log_probs(z, mask)
    p = lp.exp()
    assert p[0, 2] == 0
    torch.testing.assert_close(p.sum(-1), torch.ones(1))
    torch.testing.assert_close(p[:, :2], z[:, :2].softmax(-1))
    y = torch.tensor([1])
    torch.testing.assert_close(loss(lp, y, "ce"), -lp[0, 1])
    torch.testing.assert_close(
        loss(lp, y, "brier"), ((p - torch.tensor([[0, 1, 0]])) ** 2).sum()
    )
    perm = torch.tensor([2, 0, 1])
    torch.testing.assert_close(
        masked_log_probs(z[:, perm], mask[:, perm]).exp()[:, torch.argsort(perm)], p
    )
    with pytest.raises(ValueError):
        masked_log_probs(z, torch.zeros_like(mask))


def test_pg_semantic_permutation():
    z = torch.tensor([[0.2, -0.3, 0.7]], dtype=torch.float64, requires_grad=True)
    draws = torch.tensor([[0, 2, 2, 1]])
    y = torch.tensor([2])
    perm = torch.tensor([2, 0, 1])
    inv = perm.argsort()
    original = paired_pg(z.log_softmax(-1), y, draws=draws)
    permuted = paired_pg(z[:, perm].log_softmax(-1), inv[y], draws=inv[draws])
    torch.testing.assert_close(original, permuted)
    torch.testing.assert_close(
        torch.autograd.grad(original, z)[0], torch.autograd.grad(permuted, z)[0]
    )
