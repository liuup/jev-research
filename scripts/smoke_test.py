"""GPU smoke: model, decision head, one backward pass, then one tiny generation."""

import importlib.metadata
import json
import os
from pathlib import Path

import torch
from transformers.models.qwen3_5 import modeling_qwen3_5

from jevmath.config import load_config
from jevmath.dataset import collate
from jevmath.env import State
from jevmath.gsm8k import load_rows
from jevmath.model import JevModel, load_tokenizer
from jevmath.objectives import loss
from jevmath.rollout import score_candidates
from jevmath.serialization import question_row
from jevmath.trainer import run
from jevmath.utils import save_json, seed_all

if __name__ == "__main__":
    assert os.environ.get("SLURM_JOB_ID"), "GPU work must run under Slurm"
    torch.set_num_threads(4)
    assert modeling_qwen3_5.is_fast_path_available
    config = load_config("configs/model.yaml") | load_config("configs/smoke.yaml")
    config["policy_id"] = f"smoke:seed{config['seed']}"
    seed_all(config["seed"])
    torch.cuda.reset_peak_memory_stats()
    model = JevModel.load_base(config).cuda().train()
    names = list(dict(model.named_parameters()))
    assert not any("visual" in name or "lm_head" in name for name in names)
    assert all(parameter.requires_grad for parameter in model.backbone.parameters())
    tokenizer = load_tokenizer(config["base_model"])

    rows = load_rows(Path(config["data_dir"]) / "train.jsonl")
    state = State.initial(rows[0])
    candidates = state.candidates()
    probabilities = score_candidates(model, tokenizer, [(state, candidates)], config)[0]
    assert len(probabilities) == len(candidates)
    assert all(0.0 <= value <= 1.0 for value in probabilities.values())

    probe = question_row(state, candidates[0], "smoke") | dict(observed_outcome="yes")
    batch = collate([probe], tokenizer, config["max_length"])
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logp = model(batch)
    torch.testing.assert_close(
        logp.exp().sum(-1), torch.ones(len(batch["state_ids"]), device="cuda")
    )
    loss(logp, batch["targets"].cuda(), "ce").backward()
    for name, module in (("backbone", model.backbone), ("head", model.head)):
        missing = [key for key, value in module.named_parameters() if value.grad is None]
        assert not missing, (name, missing[:5])
        assert all(torch.isfinite(parameter.grad).all() for parameter in module.parameters())
    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()

    summary = run(config, "runs/smoke")
    report = dict(
        success=True,
        candidates=len(candidates),
        first_probabilities=list(probabilities.items())[:5],
        peak_gpu_gib=torch.cuda.max_memory_allocated() / 2**30,
        text_parameters=sum(p.numel() for p in model.backbone.parameters()),
        head_parameters=sum(p.numel() for p in model.head.parameters()),
        flash_linear_attention_version=importlib.metadata.version(
            "flash-linear-attention"
        ),
        causal_conv1d_version=importlib.metadata.version("causal-conv1d"),
        generations=summary,
    )
    save_json("results/gpu_smoke.json", report)
    print(json.dumps(report, default=str), flush=True)
