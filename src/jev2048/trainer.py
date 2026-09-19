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

from .config import load_config
from .dataset import read_rows
from .evaluation import mc_metrics, observed_metrics, plot_reliability, predict
from .model import JevModel, load_tokenizer
from .online_trainer import run_online
from .optimization import optimization_step
from .runtime import atomic_checkpoint, checkpoint, load_checkpoint, require_slurm
from .serialization import OUTCOMES as OUTCOME_IDS
from .serialization import candidates
from .training_plot import plot_training_run
from .utils import digest, save_json, seed_all


def arguments():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/train_paired_pg.yaml")
    p.add_argument("--model-config", default="configs/model.yaml")
    p.add_argument("--set", action="append", default=[], metavar="KEY=YAML_VALUE")
    p.add_argument("--output")
    p.add_argument("--initialize", action="store_true")
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()
    config = load_config(a.model_config)
    config.update(load_config(a.config))
    for item in a.set:
        key, value = item.split("=", 1)
        if key not in config:
            raise ValueError(f"Unknown config key: {key}")
        config[key] = yaml.safe_load(value)
    return a, config


def learning_rate_scale(step, steps, warmup_steps):
    if step < warmup_steps:
        return (step + 1) / max(1, warmup_steps)
    return max(0.0, (steps - step) / max(1, steps - warmup_steps))


def question_schedule(row_count, steps, effective_batch, seed, global_step_offset=0):
    required = (global_step_offset + steps) * effective_batch
    rng = np.random.default_rng(seed)
    indices = []
    while len(indices) < required:
        indices.extend(rng.permutation(row_count).tolist())
    start = global_step_offset * effective_batch
    return indices[start:required]


def evaluate_run(model, tok, config, output, plot_name=None):
    root = Path(config["data_dir"])
    output = Path(output)
    report = {}
    for split in ("test", "calibration"):
        rows = read_rows(root / f"{split}.jsonl")
        probs = predict(
            model,
            tok,
            rows,
            config["inference_questions"],
            config["max_length"],
            config["inference_precision"],
        )
        metrics = observed_metrics(probs, rows)
        save_json(output / f"{split}_metrics.json", metrics)
        save_json(
            output / f"{split}_predictions.json",
            [
                dict(
                    state_id=r["state_id"],
                    action=r["action"],
                    observed_outcome=r["observed_outcome"],
                    candidate_ids=candidates(r),
                    probabilities=p.tolist(),
                )
                for r, p in zip(rows, probs)
            ],
        )
        if split == "test":
            report.update(metrics)
            plot_reliability(metrics, plot_name or output.name)
    mcpath = root / "mc_reference.jsonl"
    if mcpath.exists():
        mc_manifest = json.loads((root / "mc_manifest.json").read_text())
        assert mc_manifest["data_manifest_sha256"] == digest(root / "manifest.json")
        assert mc_manifest["sha256"] == digest(mcpath)
        rows = read_rows(mcpath)
        probs = predict(
            model,
            tok,
            rows,
            config["inference_questions"],
            config["max_length"],
            config["inference_precision"],
        )
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
    if c.get("mode") == "online":
        run_online(
            c, Path(args.output or f"runs/online_{c['objective']}_seed{c['seed']}")
        )
        return
    if c.get("mode") != "offline":
        raise ValueError("mode must be offline or online")
    for key in (
        "steps",
        "effective_batch",
        "microbatch",
        "eval_every",
        "checkpoint_every",
        "eval_questions",
        "inference_questions",
        "plot_smooth",
    ):
        if not isinstance(c[key], int) or c[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if c["effective_batch"] % c["microbatch"]:
        raise ValueError("effective_batch must be divisible by microbatch")
    root = Path(c["data_dir"])
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["training_labels"].startswith("one observed terminal event")
    assert manifest["policy_sha256"] == digest(
        manifest["config"]["heuristic_config"]
    )
    assert manifest["policy_code_sha256"] == digest("src/jev2048/heuristic.py")
    for split, info in manifest["splits"].items():
        assert info["sha256"] == digest(root / f"{split}.jsonl")
    rows = read_rows(root / "train.jsonl")
    if manifest.get("task", "terminal_max_tile") != c.get("task", "terminal_max_tile"):
        raise ValueError("Dataset and training task differ")
    if c.get("task") == "reach_2048":
        assert c["continuation_policy_id"] == f"heuristic:{manifest['policy_sha256']}"
        for split in manifest["splits"]:
            for row in read_rows(root / f"{split}.jsonl"):
                assert row.get("task") == "reach_2048" and candidates(row) == ["yes", "no"]
                assert row["continuation_policy_id"] == c["continuation_policy_id"]
                assert max(map(max, row["board"])) < 2048
                assert row["observed_outcome"] in ("yes", "no")
    dev = read_rows(root / "dev.jsonl")[: c["eval_questions"]]
    assert all("q" not in r for r in rows)
    output = Path(args.output or f"runs/offline_{c['objective']}_seed{c['seed']}")
    output.mkdir(parents=True, exist_ok=True)
    resume_path = output / "resume.pt"
    continuation_from = c.get("continuation_from")
    if args.resume:
        if not resume_path.exists():
            raise FileNotFoundError(resume_path)
        model, saved = load_checkpoint(resume_path)
        if saved["config"].get("task") != c.get("task"):
            raise ValueError("Cannot resume a different task")
        start_phase_step = saved.get("phase_step", saved["step"])
        global_step_offset = saved.get("global_step_offset", 0)
        load_optimizer_state = True
        reset_optimizer_schedule = False
    elif continuation_from:
        if any(output.iterdir()):
            raise FileExistsError(f"Use a fresh run directory: {output}")
        model, saved = load_checkpoint(continuation_from)
        if saved["config"].get("task") != c.get("task"):
            raise ValueError("Cannot continue a different task")
        start_phase_step = 0
        global_step_offset = saved["step"]
        load_optimizer_state = True
        reset_optimizer_schedule = True
    else:
        if any(output.iterdir()):
            raise FileExistsError(f"Use a fresh run directory: {output}")
        model, saved = load_checkpoint(c["common_init"])
        if saved["kind"] != "common_initialization_no_warmup":
            raise ValueError("All offline objectives require the common initialization")
        start_phase_step = 0
        global_step_offset = 0
        load_optimizer_state = False
        reset_optimizer_schedule = False
    for key in ("base_model", "head_width", "attention_heads", "seed"):
        assert saved["config"][key] == c[key], key
    if continuation_from:
        for key in (
            "objective",
            "data_dir",
            "effective_batch",
            "max_length",
            "reward_samples",
            "baseline",
        ):
            assert saved["config"][key] == c[key], key
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
        return learning_rate_scale(step, c["steps"], c["warmup_steps"])

    if load_optimizer_state:
        optimizer.load_state_dict(saved["optimizer"])
    resumed_learning_rates = None
    if args.resume:
        resumed_learning_rates = [group["lr"] for group in optimizer.param_groups]
    if reset_optimizer_schedule:
        for group, learning_rate in zip(
            optimizer.param_groups, (c["backbone_lr"], c["head_lr"])
        ):
            group["lr"] = learning_rate
            group["initial_lr"] = learning_rate
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    if args.resume:
        scheduler.load_state_dict(saved["scheduler"])
        for group, learning_rate in zip(
            optimizer.param_groups, resumed_learning_rates
        ):
            group["lr"] = learning_rate
    if args.resume or continuation_from:
        torch.set_rng_state(saved["torch_rng"])
        torch.cuda.set_rng_state(saved["cuda_rng"])
        random.setstate(saved["python_rng"])
        np.random.set_state(saved["numpy_rng"])
    parent_step = saved.get("step") if continuation_from and not args.resume else None
    indices = question_schedule(
        len(rows),
        c["steps"],
        c["effective_batch"],
        c["seed"],
        global_step_offset,
    )
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
        global_step_offset=global_step_offset,
    )
    if continuation_from:
        meta.update(
            continuation_from=str(continuation_from),
            continuation_sha256=digest(continuation_from),
            parent_step=parent_step or global_step_offset,
        )
    if not args.resume:
        save_json(output / "resolved_config.json", meta)
        (output / "resolved_config.yaml").write_text(yaml.safe_dump(c))
    best_record_path = output / "best_dev.json"
    if args.resume and best_record_path.exists():
        best_record = json.loads(best_record_path.read_text())
    else:
        best_record = None
    selection_metric = c.get("selection_metric", "observed_brier")
    selection_mode = c.get("selection_mode", "min")
    if selection_mode not in ("min", "max"):
        raise ValueError("selection_mode must be min or max")
    del saved
    track_best = c.get("track_best", True)
    if continuation_from and not args.resume and track_best:
        initial_metrics = observed_metrics(
            predict(
                model,
                tok,
                dev,
                c["inference_questions"],
                c["max_length"],
                c["inference_precision"],
            ),
            dev,
        )
        best_record = dict(
            metric=selection_metric,
            mode=selection_mode,
            value=float(initial_metrics[selection_metric]),
            step=global_step_offset,
            phase_step=0,
            metrics=initial_metrics,
        )
        save_json(output / "dev" / f"step_{global_step_offset:06d}.json", initial_metrics)
        atomic_checkpoint(
            output / "best_checkpoint.pt",
            model,
            c,
            kind="best_dev_checkpoint",
            step=global_step_offset,
            phase_step=0,
            global_step_offset=global_step_offset,
            selection=best_record,
            metadata=meta,
        )
        save_json(best_record_path, best_record)
    torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    micro = c["microbatch"]
    with (output / "training.jsonl").open("a" if args.resume else "w") as log:
        for phase_step in range(start_phase_step, c["steps"]):
            global_step = global_step_offset + phase_step + 1
            batch_rows = [
                rows[i]
                for i in indices[
                    phase_step
                    * c["effective_batch"] : (phase_step + 1)
                    * c["effective_batch"]
                ]
            ]
            record, micro = optimization_step(
                model, tok, batch_rows, optimizer, scheduler, c, micro
            )
            record.pop("step_seconds", None)
            record.pop("questions_per_second", None)
            record["step"] = global_step
            evaluate = (
                (phase_step + 1) % c["eval_every"] == 0
                or phase_step + 1 == c["steps"]
            )
            if evaluate:
                dev_metrics = observed_metrics(
                    predict(
                        model,
                        tok,
                        dev,
                        c["inference_questions"],
                        c["max_length"],
                        c["inference_precision"],
                    ),
                    dev,
                )
                save_json(output / "dev" / f"step_{global_step:06d}.json", dev_metrics)
                record["dev"] = {
                    key: value for key, value in dev_metrics.items() if key != "events"
                }
                value = float(dev_metrics[selection_metric])
                improved = best_record is None or (
                    value < best_record["value"]
                    if selection_mode == "min"
                    else value > best_record["value"]
                )
                if improved and track_best:
                    best_record = dict(
                        metric=selection_metric,
                        mode=selection_mode,
                        value=value,
                        step=global_step,
                        phase_step=phase_step + 1,
                        metrics=dev_metrics,
                    )
                    atomic_checkpoint(
                        output / "best_checkpoint.pt",
                        model,
                        c,
                        kind="best_dev_checkpoint",
                        step=global_step,
                        phase_step=phase_step + 1,
                        global_step_offset=global_step_offset,
                        selection=best_record,
                        metadata=meta,
                    )
                    save_json(best_record_path, best_record)
            log.write(json.dumps(record) + "\n")
            log.flush()
            if evaluate and c.get("plot_every_eval", True):
                try:
                    plot_result = plot_training_run(
                        output,
                        smooth=c["plot_smooth"],
                    )
                    print(json.dumps(dict(training_plot=plot_result)), flush=True)
                except Exception as error:
                    print(
                        json.dumps(
                            dict(
                                training_plot_error=type(error).__name__,
                                message=str(error),
                                step=global_step,
                            )
                        ),
                        flush=True,
                    )
            print(
                json.dumps({k: v for k, v in record.items() if k != "dev"}), flush=True
            )
            if (phase_step + 1) % c["checkpoint_every"] == 0:
                atomic_checkpoint(
                    resume_path,
                    model,
                    c,
                    optimizer=optimizer.state_dict(),
                    scheduler=scheduler.state_dict(),
                    step=global_step,
                    phase_step=phase_step + 1,
                    global_step_offset=global_step_offset,
                    torch_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state(),
                    python_rng=random.getstate(),
                    numpy_rng=np.random.get_state(),
                    metadata=meta,
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
    if c.get("save_final_checkpoint", True):
        checkpoint(
            output / "checkpoint.pt",
            model,
            c,
            optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(),
            step=global_step_offset + c["steps"],
            phase_step=c["steps"],
            global_step_offset=global_step_offset,
            torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state(),
            python_rng=random.getstate(),
            numpy_rng=np.random.get_state(),
            metadata=meta,
        )
    del optimizer
    if not c.get("evaluate_after_training", True):
        save_json(output / "selection.json", dict(best_dev=best_record))
        print(json.dumps(dict(stats=stats, best_dev=best_record)), flush=True)
        return
    metrics = evaluate_run(model, tok, c, output, plot_name=output.name)
    best_metrics = None
    if c.get("evaluate_best", False) and (output / "best_checkpoint.pt").exists():
        del model
        torch.cuda.empty_cache()
        best_model, _ = load_checkpoint(output / "best_checkpoint.pt")
        best_metrics = evaluate_run(
            best_model,
            tok,
            c,
            output / "best_eval",
            plot_name=f"{output.name}_best",
        )
        del best_model
        torch.cuda.empty_cache()
    save_json(
        output / "selection.json",
        dict(best_dev=best_record, final_metrics=metrics, best_metrics=best_metrics),
    )
    print(
        json.dumps(dict(stats=stats, metrics=metrics, best_metrics=best_metrics)),
        flush=True,
    )
