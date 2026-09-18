import itertools
import pytest
import torch
from jev2048.objectives import loss, paired_pg, masked_log_probs

@pytest.mark.parametrize("k,m",[(2,2),(3,2),(2,3)])
@pytest.mark.parametrize("baseline",["conditional","zero"])
def test_exact_gradient(k,m,baseline):
    z=torch.linspace(-0.6,0.8,k,dtype=torch.float64,requires_grad=True)
    p=z.softmax(0); q=torch.arange(1,k+1,dtype=torch.float64); q=q/q.sum()
    expected=torch.zeros_like(z)
    reward=0.
    for draws in itertools.product(range(k),repeat=m):
        a=torch.tensor([draws]); weight=p[list(draws)].prod().detach()
        counts=torch.bincount(a[0],minlength=k)
        for y in range(k):
            l=paired_pg(z.log_softmax(0)[None],torch.tensor([y]),baseline=baseline,draws=a)
            g=torch.autograd.grad(l,z,retain_graph=True)[0]
            expected += weight*q[y]*g
            r=2/m*sum(x==y for x in draws)-(counts*(counts-1)).sum()/(m*(m-1))
            reward += weight*q[y]*r
    actual=torch.autograd.grad(((p-q)**2).sum(),z)[0]
    torch.testing.assert_close(expected,actual,atol=1e-14,rtol=1e-13)
    torch.testing.assert_close(reward,2*(p*q).sum()-(p*p).sum(),atol=1e-7,rtol=1e-7)

def test_scores_mask_permutation():
    z=torch.tensor([[1.,2.,9.]])
    mask=torch.tensor([[True,True,False]])
    lp=masked_log_probs(z,mask); p=lp.exp()
    assert p[0,2]==0
    torch.testing.assert_close(p.sum(-1),torch.ones(1))
    torch.testing.assert_close(p[:,:2],z[:,:2].softmax(-1))
    y=torch.tensor([1])
    torch.testing.assert_close(loss(lp,y,"ce"),-lp[0,1])
    torch.testing.assert_close(loss(lp,y,"brier"),((p-torch.tensor([[0,1,0]]))**2).sum())
    perm=torch.tensor([2,0,1])
    torch.testing.assert_close(masked_log_probs(z[:,perm],mask[:,perm]).exp()[:,torch.argsort(perm)],p)
    with pytest.raises(ValueError): masked_log_probs(z,torch.zeros_like(mask))
