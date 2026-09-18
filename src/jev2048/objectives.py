"""Proper scores and RLCD-inspired paired reward (not official TypeSafe RLCD)."""
import torch
import torch.nn.functional as F

def masked_log_probs(logits, mask):
    if not mask.any(-1).all(): raise ValueError("Every question needs a valid candidate")
    return F.log_softmax(logits.float().masked_fill(~mask,-torch.inf),dim=-1)

def paired_pg(logp, targets, samples=32, baseline="conditional", draws=None):
    p = logp.exp()
    if draws is None: draws = torch.multinomial(p,samples,replacement=True)
    m=draws.shape[1]
    if m<2: raise ValueError("M must be >=2")
    with torch.no_grad():
        counts = torch.zeros_like(p).scatter_add_(1,draws,torch.ones_like(draws,dtype=p.dtype))
        rewards = 2/m*(draws==targets[:,None]).to(p.dtype)-2*(counts.gather(1,draws)-1)/(m*(m-1))
        if baseline == "conditional":
            drawn_p = p.gather(1,draws)
            b = 2/m*p.gather(1,targets[:,None])-2*(drawn_p.sum(1,keepdim=True)-drawn_p)/(m*(m-1))
        elif baseline == "zero": b=0
        else: raise ValueError(baseline)
        advantage = rewards-b
    return -(advantage*logp.gather(1,draws)).sum(1).mean()

def loss(logp, targets, objective, samples=32, baseline="conditional"):
    if objective == "ce": return -logp.gather(1,targets[:,None]).mean()
    if objective == "brier": return ((logp.exp()-F.one_hot(targets,logp.shape[-1]))**2).sum(-1).mean()
    if objective == "paired_pg": return paired_pg(logp,targets,samples,baseline)
    raise ValueError(objective)
