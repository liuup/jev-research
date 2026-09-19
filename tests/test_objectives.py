import itertools
import unittest

import torch

from jevsnake.objectives import masked_log_probs, paired_pg


class ObjectiveTests(unittest.TestCase):
    def test_masked_log_probs_normalizes_only_valid_candidates(self):
        logits = torch.tensor([[1.0, 2.0, 99.0]])
        mask = torch.tensor([[True, True, False]])
        logp = masked_log_probs(logits, mask)
        self.assertTrue(torch.allclose(logp[:, :2].exp().sum(-1), torch.ones(1)))
        self.assertTrue(torch.isneginf(logp[0, 2]))

    def test_expected_paired_pg_gradient_matches_brier_gradient(self):
        logits = torch.tensor([[0.2, -0.1, 0.4]], dtype=torch.float64, requires_grad=True)
        logp = logits.log_softmax(-1)
        target = torch.tensor([1])
        probabilities = logp.detach().exp()[0]
        expected_gradient = torch.zeros_like(logits)
        for draw in itertools.product(range(3), repeat=2):
            weight = probabilities[draw[0]] * probabilities[draw[1]]
            objective = paired_pg(
                logp,
                target,
                draws=torch.tensor([draw]),
                baseline="conditional",
            )
            gradient = torch.autograd.grad(objective, logits, retain_graph=True)[0]
            expected_gradient += weight * gradient
        brier = ((logp.exp() - torch.nn.functional.one_hot(target, 3)) ** 2).sum()
        reference = torch.autograd.grad(brier, logits)[0]
        self.assertTrue(torch.allclose(expected_gradient, reference, atol=1e-12, rtol=1e-12))
