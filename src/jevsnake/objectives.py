"""Proper scores and an RLCD-inspired paired proper-reward estimator."""

import torch
import torch.nn.functional as F


def masked_log_probs(logits, mask):
    if not mask.any(-1).all():
        raise ValueError("Every question needs at least one valid candidate")
    return F.log_softmax(logits.float().masked_fill(~mask, -torch.inf), dim=-1)


def paired_pg(
    logp, targets, samples=32, baseline="conditional", draws=None, diagnostics=None
):
    p = logp.exp()
    if draws is None:
        draws = torch.multinomial(p, samples, replacement=True)
    m = draws.shape[1]
    if m < 2:
        raise ValueError("paired_pg requires at least two samples")
    with torch.no_grad():
        counts = torch.zeros_like(p).scatter_add_(
            1, draws, torch.ones_like(draws, dtype=p.dtype)
        )
        rewards = 2 / m * (draws == targets[:, None]).to(p.dtype) - 2 * (
            counts.gather(1, draws) - 1
        ) / (m * (m - 1))
        if baseline == "conditional":
            drawn_p = p.gather(1, draws)
            control = 2 / m * p.gather(1, targets[:, None]) - 2 * (
                drawn_p.sum(1, keepdim=True) - drawn_p
            ) / (m * (m - 1))
        elif baseline == "zero":
            control = torch.zeros_like(rewards)
        else:
            raise ValueError("baseline must be conditional or zero")
        advantage = rewards - control
        if diagnostics is not None:
            hits = (draws == targets[:, None]).to(p.dtype).mean(1)
            collisions = (counts * (counts - 1)).sum(1) / (m * (m - 1))
            diagnostics.update(
                sampled_reward=(2 * hits - collisions).mean(),
                sampled_hit_rate=hits.mean(),
                sampled_collision_rate=collisions.mean(),
                advantage_mean=advantage.mean(),
                advantage_second_moment=advantage.square().mean(),
                advantage_abs_mean=advantage.abs().mean(),
                advantage_positive_fraction=(advantage > 0).to(p.dtype).mean(),
                baseline_mean=control.mean(),
            )
    return -(advantage * logp.gather(1, draws)).sum(1).mean()


def loss(logp, targets, objective, samples=32, baseline="conditional", diagnostics=None):
    if objective == "ce":
        return -logp.gather(1, targets[:, None]).mean()
    if objective == "brier":
        onehot = F.one_hot(targets, logp.shape[-1]).to(logp.dtype)
        return ((logp.exp() - onehot) ** 2).sum(-1).mean()
    if objective == "paired_pg":
        return paired_pg(logp, targets, samples, baseline, diagnostics=diagnostics)
    raise ValueError(f"Unknown objective: {objective}")
