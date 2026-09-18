import argparse
import hashlib
import json
import os
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from .dataset import collate, read_rows
from .evaluation import mc_metrics, observed_metrics, plot_reliability, predict
from .model import JevModel, load_tokenizer
from .objectives import loss
from .serialization import OUTCOMES as OUTCOME_IDS
from .utils import digest, save_json, seed_all


def arguments():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/train_ce.yaml")
    p.add_argument("--model-config", default="configs/model.yaml")
    p.add_argument("--set", action="append", default=[], metavar="KEY=YAML_VALUE")
    p.add_argument("--output")
    p.add_argument("--initialize", action="store_true")
    a = p.parse_args()
    config = yaml.safe_load(open(a.model_config))
    config.update(yaml.safe_load(open(a.config)))
    for item in a.set:
        key, value = item.split("=", 1)
        if key not in config:
            raise ValueError(f"Unknown config key: {key}")
        config[key] = yaml.safe_load(value)
    return a, config


def require_slurm():
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("GPU work requires a Slurm allocation")
    torch.set_num_threads(4)


def checkpoint(path, model, config, **extra):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(model=model.state_dict(), config=config, **extra), path)


def load_checkpoint(path):
    saved = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    model = JevModel.load_base(saved["config"]).float()
    model.load_state_dict(saved["model"], strict=True)
    return model.cuda(), saved


def evaluate_run(model, tok, config, output):
    root = Path(config["data_dir"])
    output = Path(output)
    report = {}
    for split in ("test", "calibration"):
        rows = read_rows(root / f"{split}.jsonl")
        probs = predict(model, tok, rows, config["microbatch"], config["max_length"])
        metrics = observed_metrics(probs, rows)
        save_json(output / f"{split}_metrics.json", metrics)
        save_json(
            output / f"{split}_predictions.json",
            [
                dict(
                    state_id=r["state_id"],
                    action=r["action"],
                    observed_outcome=r["observed_outcome"],
                    candidate_ids=OUTCOME_IDS,
                    probabilities=p.tolist(),
                )
                for r, p in zip(rows, probs)
            ],
        )
        if split == "test":
            report.update(metrics)
            plot_reliability(metrics, output.name)
    mcpath = root / "mc_reference.jsonl"
    if mcpath.exists():
        mc_manifest = json.loads((root / "mc_manifest.json").read_text())
        assert mc_manifest["data_manifest_sha256"] == digest(root / "manifest.json")
        assert mc_manifest["sha256"] == digest(mcpath)
        rows = read_rows(mcpath)
        probs = predict(model, tok, rows, config["microbatch"], config["max_length"])
        report.update(mc_metrics(probs, rows))
        save_json(
            output / "mc_predictions.json",
            [
                dict(
                    state_id=r["state_id"],
                    action=r["action"],
                    probabilities=p.tolist(),
                    q=r["q"],
                )
                for r, p in zip(rows, probs)
            ],
        )
    save_json(output / "metrics.json", report)
    return report


def main():
    args, c = arguments()
    require_slurm()
    print(yaml.safe_dump(c), flush=True)
    seed_all(c["seed"])
    if args.initialize:
        path = Path(c["common_init"])
        if path.exists():
            raise FileExistsError("Refusing to overwrite common initialization")
        model = JevModel.load_base(c).float()
        checkpoint(
            path, model, c, kind="common_initialization_no_warmup", seed=c["seed"]
        )
        print("Common initialization:", path, digest(path), flush=True)
        return
    root = Path(c["data_dir"])
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["policy_sha256"] == digest("configs/heuristic.yaml")
    assert manifest["policy_code_sha256"] == digest("src/jev2048/heuristic.py")
    for split, info in manifest["splits"].items():
        assert info["sha256"] == digest(root / f"{split}.jsonl")
    rows = read_rows(root / "train.jsonl")
    dev = read_rows(root / "dev.jsonl")[: c["eval_questions"]]
    assert all("q" not in r for r in rows)
    output = Path(args.output or f"runs/{c['objective']}_seed{c['seed']}")
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(f"Use a fresh run directory: {output}")
    model, saved = load_checkpoint(c["common_init"])
    assert saved["kind"] == "common_initialization_no_warmup"
    for key in ("base_model", "head_width", "attention_heads", "seed"):
        assert saved["config"][key] == c[key], key
    del saved
    tok = load_tokenizer(c["base_model"])
    optimizer = torch.optim.AdamW(
        [
            dict(params=model.backbone.parameters(), lr=c["backbone_lr"]),
            dict(params=model.head.parameters(), lr=c["head_lr"]),
        ],
        weight_decay=c["weight_decay"],
        foreach=False,
    )

    def schedule(step):
        if step < c["warmup_steps"]:
            return (step + 1) / max(1, c["warmup_steps"])
        return max(0.0, (c["steps"] - step) / max(1, c["steps"] - c["warmup_steps"]))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    rng = np.random.default_rng(c["seed"])
    indices = []
    while len(indices) < c["steps"] * c["effective_batch"]:
        indices.extend(rng.permutation(len(rows)).tolist())
    indices = indices[: c["steps"] * c["effective_batch"]]
    meta = dict(
        config=c,
        git_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        git_dirty=bool(
            subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        ),
        versions=dict(torch=torch.__version__),
        data_manifest_sha256=digest(root / "manifest.json"),
        common_init_sha256=digest(c["common_init"]),
        batch_schedule_sha256=hashlib.sha256(
            np.array(indices, dtype=np.int64).tobytes()
        ).hexdigest(),
        slurm_job_id=os.environ["SLURM_JOB_ID"],
    )
    save_json(output / "resolved_config.json", meta)
    (output / "resolved_config.yaml").write_text(yaml.safe_dump(c))
    torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    micro = c["microbatch"]
    with (output / "training.jsonl").open("w") as log:
        for step in range(c["steps"]):
            step_start = time.monotonic()
            batch_rows = [
                rows[i]
                for i in indices[
                    step * c["effective_batch"] : (step + 1) * c["effective_batch"]
                ]
            ]
            cpu_rng = torch.get_rng_state()
            cuda_rng = torch.cuda.get_rng_state()
            while True:
                try:
                    model.train()
                    optimizer.zero_grad(set_to_none=True)
                    total = 0.0
                    metrics = {}
                    predicted_mass = torch.zeros(len(OUTCOME_IDS), device="cuda")
                    observed_counts = torch.zeros_like(predicted_mass)
                    for offset in range(0, len(batch_rows), micro):
                        subset = batch_rows[offset : offset + micro]
                        batch = collate(subset, tok, c["max_length"])
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            lp = model(batch)
                        targets = batch["targets"].cuda()
                        diagnostics = {}
                        l = (
                            loss(
                                lp,
                                targets,
                                c["objective"],
                                c["reward_samples"],
                                c["baseline"],
                                diagnostics=diagnostics,
                            )
                            * len(subset)
                            / len(batch_rows)
                        )
                        l.backward()
                        total += float(l.detach())
                        with torch.no_grad():
                            p = lp.detach().exp()
                            py = p.gather(1, targets[:, None]).squeeze(1)
                            diagnostics.update(
                                observed_nll=-lp.detach()
                                .gather(1, targets[:, None])
                                .mean(),
                                observed_brier=(p.square().sum(1) - 2 * py + 1).mean(),
                                expected_reward_given_y=(
                                    2 * py - p.square().sum(1)
                                ).mean(),
                                target_probability=py.mean(),
                                entropy=-(
                                    p * lp.detach().masked_fill(~torch.isfinite(lp), 0)
                                )
                                .sum(1)
                                .mean(),
                                top1_accuracy=(p.argmax(1) == targets).float().mean(),
                                max_probability=p.max(1).values.mean(),
                            )
                            for key, value in diagnostics.items():
                                metrics[key] = metrics.get(
                                    key, 0
                                ) + value.detach() * len(subset) / len(batch_rows)
                            predicted_mass += p.sum(0)
                            observed_counts += torch.bincount(
                                targets, minlength=len(OUTCOME_IDS)
                            )
                    break
                except torch.cuda.OutOfMemoryError:
                    if micro == 1:
                        raise
                    optimizer.zero_grad(set_to_none=True)
                    # Release graph references before retrying the entire logical batch.
                    lp = None
                    l = None
                    batch = None
                    micro = max(1, micro // 2)
                    torch.cuda.empty_cache()
                    torch.set_rng_state(cpu_rng)
                    torch.cuda.set_rng_state(cuda_rng)
                    print(
                        f"OOM: retrying logical batch with microbatch={micro}",
                        flush=True,
                    )
            # An optimizer OOM must fail, not retry after a potentially partial update.
            norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0, error_if_nonfinite=True
            )
            learning_rates = [group["lr"] for group in optimizer.param_groups]
            optimizer.step()
            scheduler.step()
            metrics = {key: float(value) for key, value in metrics.items()}
            if "advantage_second_moment" in metrics:
                metrics["advantage_std"] = (
                    max(
                        0,
                        metrics.pop("advantage_second_moment")
                        - metrics["advantage_mean"] ** 2,
                    )
                    ** 0.5
                )
            step_seconds = time.monotonic() - step_start
            record = dict(
                step=step + 1,
                loss=total,
                gradient_norm=float(norm),
                gradient_clip_scale=min(1.0, 1.0 / (float(norm) + 1e-6)),
                microbatch=micro,
                gradient_accumulation=(len(batch_rows) + micro - 1) // micro,
                backbone_lr=learning_rates[0],
                head_lr=learning_rates[1],
                step_seconds=step_seconds,
                questions_per_second=len(batch_rows) / step_seconds,
                elapsed_seconds=time.monotonic() - start,
                train=metrics,
                mean_predicted_distribution=dict(
                    zip(OUTCOME_IDS, (predicted_mass / len(batch_rows)).tolist())
                ),
                observed_outcome_counts=dict(
                    zip(OUTCOME_IDS, observed_counts.int().tolist())
                ),
            )
            if (step + 1) % c["eval_every"] == 0 or step + 1 == c["steps"]:
                record["dev"] = observed_metrics(
                    predict(model, tok, dev, micro, c["max_length"]), dev
                )
            log.write(json.dumps(record) + "\n")
            log.flush()
            print(
                json.dumps({k: v for k, v in record.items() if k != "dev"}), flush=True
            )
    training_seconds = time.monotonic() - start
    stats = dict(
        training_seconds=training_seconds,
        peak_gpu_gib=torch.cuda.max_memory_allocated() / 2**30,
        peak_gpu_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
        actual_microbatch=micro,
    )
    save_json(output / "training_stats.json", stats)
    c["microbatch"] = micro
    checkpoint(
        output / "checkpoint.pt",
        model,
        c,
        optimizer=optimizer.state_dict(),
        scheduler=scheduler.state_dict(),
        step=c["steps"],
        torch_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state(),
        python_rng=random.getstate(),
        numpy_rng=np.random.get_state(),
        metadata=meta,
    )
    del optimizer
    metrics = evaluate_run(model, tok, c, output)
    print(json.dumps(dict(stats=stats, metrics=metrics)), flush=True)
