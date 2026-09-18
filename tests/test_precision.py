import pytest
import torch

from jev2048.evaluation import inference_precision, predict


def test_precision_context_restores_flags_on_error():
    before = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    with pytest.raises(RuntimeError):
        with inference_precision("fp32"):
            assert not torch.backends.cuda.matmul.allow_tf32
            assert not torch.backends.cudnn.allow_tf32
            raise RuntimeError("test")
    assert before == (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    with pytest.raises(ValueError):
        with inference_precision("invalid"):
            pass


def test_fp32_rejects_reduced_precision_weights():
    model = torch.nn.Linear(2, 2).bfloat16()
    with pytest.raises(ValueError, match="FP32 model parameters"):
        predict(model, None, [])
