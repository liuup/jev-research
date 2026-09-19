"""Probe batch sizes against GPU memory and throughput to pick the run defaults.

Measures the three shapes that matter:

* rollout scoring: ``inference_questions`` candidate questions per forward,
* a training step: ``microbatch`` questions per backward with gradient accumulation,
* padded prompt length and the real token volume behind each number.

The largest setting inside the memory budget wins; every measurement is written to
``results/benchmark_batches.json`` together with the OOM boundaries.
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch

from jevmath.config import load_config
from jevmath.dataset import collate
from jevmath.env import State
from jevmath.evaluation import inference_precision
from jevmath.gsm8k import load_rows
from jevmath.model import JevModel, load_tokenizer
from jevmath.objectives import loss
from jevmath.optimization import optimization_step
from jevmath.rollout import score_candidates
from jevmath.runtime import require_slurm
from jevmath.serialization import question_row
from jevmath.utils import save_json, seed_all


def build_questions(states, tokenizer, config, count):
    """Real candidate questions, in environment order, until ``count`` is reached."""
    rows = []
    for state in states:
        for candidate in state.candidates():
            rows.append(question_row(state, candidate, "benchmark"))
            if len(rows) >= count:
                return rows
    raise ValueError(f"Only {len(rows)} candidate questions available")


def peak_gib():
    return torch.cuda.max_memory_allocated() / 2**30


def measure_forward(model, tokenizer, rows, config, precision, repeats=3):
    batch = collate(rows, tokenizer, config["max_length"])
    tokens = int(batch["tokens"]["input_ids"].numel())
    lengths = batch["tokens"]["attention_mask"].sum(-1)
    model.eval()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    seconds = []
    with torch.no_grad(), inference_precision(precision):
        for _ in range(repeats):
            started = time.perf_counter()
            with torch.autocast(
                "cuda", dtype=torch.bfloat16, enabled=precision == "bf16"
            ):
                model(batch)
            torch.cuda.synchronize()
            seconds.append(time.perf_counter() - started)
    best = min(seconds)
    return dict(
        mode="forward",
        questions=len(rows),
        sequences=int(batch["tokens"]["input_ids"].shape[0]),
        tokens=tokens,
        padded_length=int(batch["tokens"]["input_ids"].shape[1]),
        mean_question_tokens=float(lengths.float().mean()),
        peak_gib=peak_gib(),
        seconds=best,
        questions_per_second=len(rows) / best,
        tokens_per_second=tokens / best,
    )


def measure_training_step(
    model, tokenizer, rows, config, optimizer, scheduler, micro, objective, repeats=3
):
    config = dict(config, objective=objective, microbatch=micro)
    model.train()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    seconds = []
    for attempt in range(repeats + 1):
        record, actual_micro = optimization_step(
            model, tokenizer, rows, optimizer, scheduler, config, micro
        )
        if attempt:
            seconds.append(record["step_seconds"])
    best = min(seconds)
    return dict(
        mode="train",
        objective=objective,
        microbatch=actual_micro,
        rows=len(rows),
        gradient_accumulation=record["gradient_accumulation"],
        peak_gib=peak_gib(),
        seconds=best,
        questions_per_second=len(rows) / best,
        gradient_norm=record["gradient_norm"],
    )


def measure_rollout(model, tokenizer, states, config, count):
    """Whole-state scoring exactly as collection calls it, including tokenization."""
    entries = [(state, state.candidates()) for state in states[:count]]
    questions = sum(len(candidates) for _, candidates in entries)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    score_candidates(model, tokenizer, entries, config)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    return dict(
        mode="rollout",
        states=len(entries),
        questions=questions,
        peak_gib=peak_gib(),
        seconds=seconds,
        states_per_second=len(entries) / seconds,
        questions_per_second=questions / seconds,
    )


def sweep(label, function, sizes, oom, budget):
    rows = []
    for size in sizes:
        try:
            measurement = function(size)
        except torch.cuda.OutOfMemoryError:
            oom.append(dict(section=label, size=size))
            torch.cuda.empty_cache()
            print(json.dumps(dict(section=label, size=size, result="oom")), flush=True)
            break
        measurement["size"] = size
        measurement["within_budget"] = measurement["peak_gib"] <= budget
        rows.append(measurement)
        print(json.dumps(dict(section=label, **measurement)), flush=True)
    return rows


if __name__ == "__main__":
    assert os.environ.get("SLURM_JOB_ID"), "GPU work must run under Slurm"
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--problems", type=int, default=24)
    parser.add_argument("--budget-gib", type=float, default=26.0)
    parser.add_argument("--forward-sizes", type=int, nargs="+", default=[16, 32, 64, 128, 256, 512])
    parser.add_argument("--train-sizes", type=int, nargs="+", default=[4, 8, 16, 32, 64, 128])
    parser.add_argument("--rollout-states", type=int, nargs="+", default=[4, 8, 16, 24])
    parser.add_argument("--effective-batch", type=int, default=64)
    parser.add_argument("--output", default="results/benchmark_batches.json")
    args = parser.parse_args()
    require_slurm()
    torch.set_num_threads(4)
    config = load_config("configs/model.yaml") | load_config(args.config)
    config["policy_id"] = "benchmark"
    seed_all(config["seed"])
    model = JevModel.load_base(config).cuda()
    tokenizer = load_tokenizer(config["base_model"])
    problems = [row for row in load_rows(Path(config["data_dir"]) / "train.jsonl") if len(row["numbers"]) >= 3][: args.problems]
    states = [State.initial(row) for row in problems]
    sizes = [len(state.quantities) for state in states]
    candidates = [len(state.candidates()) for state in states]
    questions = build_questions(states, tokenizer, config, max(args.forward_sizes))
    optimizer = torch.optim.AdamW(
        [
            dict(params=model.backbone.parameters(), lr=1e-8),
            dict(params=model.head.parameters(), lr=1e-7),
        ],
        foreach=False,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    train_rows = [dict(row, observed_outcome="yes" if index % 2 else "no") for index, row in enumerate(questions[: max(args.train_sizes)])]
    report = dict(
        config=dict(
            max_length=config["max_length"],
            problems=len(problems),
            pool_sizes=sizes,
            candidates_per_state=candidates,
            effective_batch=args.effective_batch,
            budget_gib=args.budget_gib,
        ),
        model=dict(
            text_parameters=sum(p.numel() for p in model.backbone.parameters()),
            head_parameters=sum(p.numel() for p in model.head.parameters()),
        ),
        sections={},
        oom=[],
    )
    for precision in ("fp32", "bf16"):
        label = f"forward_{precision}"
        report["sections"][label] = sweep(
            label,
            lambda size, precision=precision: measure_forward(
                model, tokenizer, questions[:size], config, precision
            ),
            args.forward_sizes,
            report["oom"],
            args.budget_gib,
        )
    report["sections"]["rollout"] = sweep(
        "rollout",
        lambda count: measure_rollout(model, tokenizer, states, config, count),
        [size for size in args.rollout_states if size <= len(states)],
        report["oom"],
        args.budget_gib,
    )
    for objective in ("paired_pg", "ce"):
        label = f"train_{objective}"
        report["sections"][label] = sweep(
            label,
            lambda size, objective=objective: measure_training_step(
                model,
                tokenizer,
                train_rows,
                config,
                optimizer,
                scheduler,
                size,
                objective,
            ),
            [size for size in args.train_sizes if size <= len(train_rows)],
            report["oom"],
            args.budget_gib,
        )
    model.backbone.gradient_checkpointing_disable()
    report["sections"]["train_ce_no_checkpointing"] = sweep(
        "train_ce_no_checkpointing",
        lambda size: measure_training_step(
            model, tokenizer, train_rows, config, optimizer, scheduler, size, "ce"
        ),
        [size for size in args.train_sizes if size <= len(train_rows)],
        report["oom"],
        args.budget_gib,
    )
    model.backbone.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    report["recommendation"] = {}
    for label, rows in report["sections"].items():
        inside = [row for row in rows if row["within_budget"]]
        if not inside:
            report["recommendation"][label] = None
            continue
        best = max(inside, key=lambda row: row["questions_per_second"])
        largest = max(inside, key=lambda row: row["size"])
        report["recommendation"][label] = dict(
            throughput_choice=dict(size=best["size"], questions_per_second=best["questions_per_second"], peak_gib=best["peak_gib"]),
            largest_inside_budget=dict(size=largest["size"], questions_per_second=largest["questions_per_second"], peak_gib=largest["peak_gib"]),
        )
    save_json(args.output, report)
    print(json.dumps(dict(recommendation=report["recommendation"], oom=report["oom"]), indent=2), flush=True)
