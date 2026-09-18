import json
import os

import torch
import yaml

from jev2048.dataset import collate, read_rows
from jev2048.model import JevModel, load_tokenizer
from jev2048.objectives import loss
from jev2048.utils import save_json, seed_all

if __name__ == "__main__":
    assert os.environ.get("SLURM_JOB_ID"), "GPU work must run under Slurm"
    torch.set_num_threads(4)
    config = yaml.safe_load(open("configs/model.yaml"))
    print(config, flush=True)
    seed_all(config["seed"])
    torch.cuda.reset_peak_memory_stats()
    model = JevModel.load_base(config).cuda().train()
    names = list(dict(model.named_parameters()))
    assert not any("visual" in n or "lm_head" in n for n in names)
    assert all(p.requires_grad for p in model.backbone.parameters())
    tok = load_tokenizer(config["base_model"])
    batch = collate(read_rows("data/smoke/train.jsonl")[:1], tok, config["max_length"])
    with torch.autocast("cuda", dtype=torch.bfloat16):
        lp = model(batch)
    torch.testing.assert_close(lp.exp().sum(-1), torch.ones(1, device="cuda"))
    loss(lp, batch["targets"].cuda(), "ce").backward()
    for name, module in [("backbone", model.backbone), ("head", model.head)]:
        missing = [n for n, p in module.named_parameters() if p.grad is None]
        assert not missing, (name, missing)
        assert all(torch.isfinite(p.grad).all() for p in module.parameters())
    report = dict(
        success=True,
        peak_gpu_gib=torch.cuda.max_memory_allocated() / 2**30,
        text_parameters=sum(p.numel() for p in model.backbone.parameters()),
        head_parameters=sum(p.numel() for p in model.head.parameters()),
        vision_parameters=0,
        vision_excluded=True,
        lm_head_excluded=True,
        probabilities=lp.exp().detach().cpu().tolist(),
    )
    save_json("results/gpu_smoke.json", report)
    print(json.dumps(report), flush=True)
