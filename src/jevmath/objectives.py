"""Proper scores and the NanoJev-style paired proper-reward policy gradient.

For draws ``A_i ~ p`` with replacement and one observed outcome ``Y``:

    R      = (2/M) sum_i 1[A_i=Y] - sum_k c[k](c[k]-1)/(M(M-1))
    E[R]   = 2 p.q - ||p||^2 = ||q||^2 - ||p-q||^2
    b_i    = (2/M) p[Y] - 2 sum_{j!=i} p[A_j]/(M(M-1))
    L_PG   = -sum_i stop_gradient(r_i - b_i) log p[A_i]
    E[grad L_PG] = grad ||p - q||^2

``Y`` must be independent of the predictive draws given the question, and the baseline
must not depend on ``A_i`` itself.
"""

import torch
import torch.nn.functional as F


def masked_log_probs(logits, mask):
    if not mask.any(-1).all():
        raise ValueError("Every question needs a valid candidate")
    return F.log_softmax(logits.float().masked_fill(~mask, -torch.inf), dim=-1)


def paired_pg(
    logp, targets, samples=32, baseline="conditional", draws=None, diagnostics=None
):
    p = logp.exp()
    if draws is None:
        draws = torch.multinomial(p, samples, replacement=True)
    m = draws.shape[1]
    if m < 2:
        raise ValueError("M must be >= 2")
    with torch.no_grad():
        counts = torch.zeros_like(p).scatter_add_(
            1, draws, torch.ones_like(draws, dtype=p.dtype)
        )
        rewards = 2 / m * (draws == targets[:, None]).to(p.dtype) - 2 * (
            counts.gather(1, draws) - 1
        ) / (m * (m - 1))
        if baseline == "conditional":
            drawn_p = p.gather(1, draws)
            b = 2 / m * p.gather(1, targets[:, None]) - 2 * (
                drawn_p.sum(1, keepdim=True) - drawn_p
            ) / (m * (m - 1))
        elif baseline == "zero":
            b = 0
        else:
            raise ValueError(baseline)
        advantage = rewards - b
        if diagnostics is not None:
            hit_rate = (draws == targets[:, None]).to(p.dtype).mean(1)
            collision = (counts * (counts - 1)).sum(1) / (m * (m - 1))
            diagnostics.update(
                sampled_reward=(2 * hit_rate - collision).mean(),
                sampled_hit_rate=hit_rate.mean(),
                sampled_collision_rate=collision.mean(),
                advantage_mean=advantage.mean(),
                advantage_second_moment=advantage.square().mean(),
                advantage_abs_mean=advantage.abs().mean(),
                advantage_positive_fraction=(advantage > 0).to(p.dtype).mean(),
                baseline_mean=torch.as_tensor(b, device=p.device, dtype=p.dtype).mean(),
                per_sample_reward_mean=rewards.mean(),
            )
    return -(advantage * logp.gather(1, draws)).sum(1).mean()


def loss(logp, targets, objective, samples=32, baseline="conditional", diagnostics=None):
    if bool((targets < 0).any()):
        raise ValueError("Training rows need an observed outcome")
    if objective == "ce":
        return -logp.gather(1, targets[:, None]).mean()
    if objective == "brier":
        return ((logp.exp() - F.one_hot(targets, logp.shape[-1])) ** 2).sum(-1).mean()
    if objective == "paired_pg":
        return paired_pg(logp, targets, samples, baseline, diagnostics=diagnostics)
    raise ValueError(objective)
