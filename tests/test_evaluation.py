import numpy as np
import pytest
from jev2048.evaluation import observed_metrics,mc_metrics,reliability

def test_perfect_predictions():
    p=np.eye(7)
    from jev2048.serialization import OUTCOMES
    rows=[dict(observed_outcome=o,q=p[i].tolist(),rollouts=256) for i,o in enumerate(OUTCOMES)]
    result=observed_metrics(p,rows)
    assert result["top1_accuracy"]==1 and result["observed_nll"]==0 and result["observed_brier"]==0
    assert result["ece_ge2048"]==0
    assert mc_metrics(p,rows)["mc_squared_l2"]==0
    assert mc_metrics(p,rows)["mc_js"]==0

def test_bin_boundaries():
    ece,rows=reliability(np.array([0.,0.1,1.]),np.array([0.,1.,1.]))
    assert [rows[i]["count"] for i in (0,1,9)]==[1,1,1]
    assert ece==pytest.approx(.3)
