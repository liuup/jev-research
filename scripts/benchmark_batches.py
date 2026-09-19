"""Probe batch sizes against GPU memory and throughput for the 24 game.

Three shapes matter and they are measured on real puzzle states:

* forward scoring: ``inference_questions`` candidate questions per forward,
* whole-state scoring: exactly what collection calls, including tokenization,
* a training step: ``microbatch`` questions per backward with gradient accumulation.

Every measurement lands in ``results/benchmark_batches.json`` together with the OOM list.
"""

import argparse
import json
import time
from pathlib import Path

import torch

from jevmath.config import load_config
from jevmath.dataset import collate
from jevmath.env import State
from jevmath.evaluation import inference_precision
from jevmath.model import JevModel, load_tokenizer
from jevmath.optimization import optimization_step
from jevmath.puzzles import load_puzzles
from jevmath.rollout import score_candidates
from jevmath.runtime import require_slurm
from jevmath.serialization import question_row
from jevmath.utils import save_json, seed_all


def states_of(puzzles, count):
    states = [State.initial(puzzle) for puzzle in puzzles[:count]]
    return [(state, state.actions()) for state in states]


def build_questions(puzzles, policy_id, count):
    """Candidate questions in environment order, as many as the puzzles can supply."""
    rows = []
    for puzzle in puzzles:
        state = State.initial(puzzle)
        for action in state.actions():
            rows.append(question_row(state, action, policy_id))
            if len(rows) >= count:
                return rows
    return rows


def peak_gib():
    return torch.cuda.max_memory_allocated() / 2**30


def measure_forward(model, tokenizer, rows, config, precision, repeats=3):
    batch = collate(rows, tokenizer, config["max_length"])
    tokens = int(batch["tokens"]["input_ids"].numel())
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
        peak_gib=peak_gib(),
        seconds=best,
        questions_per_second=len(rows) / best,
        tokens_per_second=tokens / best,
    )


def measure_rollout(model, tokenizer, entries, config, count):
    chosen = entries[:count]
    questions = sum(len(actions) for _, actions in chosen)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    score_candidates(model, tokenizer, chosen, config)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    return dict(
        mode="rollout",
        states=count,
        questions=questions,
        peak_gib=peak_gib(),
        seconds=seconds,
        states_per_second=count / seconds,
        questions_per_second=questions / seconds,
    )


def measure_training_step(
    model, tokenizer, rows, config, optimizer, scheduler, micro, objective, repeats=3
):
    config = dict(config, objective=objective, microbatch=micro)
    model.train()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    seconds, record = [], None
    for attempt in range(repeats + 1):
        record, actual = optimization_step(
            model, tokenizer, rows, optimizer, scheduler, config, micro
        )
        if attempt:
            seconds.append(record["step_seconds"])
    best = min(seconds)
    return dict(
        mode="train",
        objective=objective,
        microbatch=record["microbatch"],
        requested_microbatch=micro,
        rows=len(rows),
        gradient_accumulation=record["gradient_accumulation"],
        peak_gib=peak_gib(),
        seconds=best,
        rows_per_second=len(rows) / best,
    )


def sweep(name, sizes, measure, budget_gib, oom, report):
    rows = []
    for size in sizes:
        try:
            row = measure(size)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            oom.append(dict(section=name, size=size))
            print(json.dumps(dict(section=name, size=size, oom=True)), flush=True)
            continue
        row["size"] = size
        rows.append(row)
        print(json.dumps(dict(section=name, **row)), flush=True)
    report[name] = rows
    inside = [row for row in rows if row["peak_gib"] <= budget_gib]
    return inside


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/24game")
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--puzzles", type=int, default=24)
    parser.add_argument("--forward-sizes", type=int, nargs="+", default=[16, 32, 64, 128, 256, 512])
    parser.add_argument("--rollout-states", type=int, nargs="+", default=[4, 8, 16, 24])
    parser.add_argument("--train-sizes", type=int, nargs="+", default=[4, 8, 16, 32, 64, 128])
    parser.add_argument("--effective-batch", type=int, default=64)
    parser.add_argument("--budget-gib", type=float, default=26.0)
    parser.add_argument("--output", default="results/benchmark_batches.json")
    args = parser.parse_args()
    require_slurm()

    config = load_config("configs/model.yaml") | load_config(args.config)
    config["policy_id"] = "benchmark"
    seed_all(config["seed"])
    model = JevModel.load_base(config).float().cuda()
    tokenizer = load_tokenizer(config["base_model"])
    available = load_puzzles(Path(args.data_dir) / "train.jsonl")
    puzzles = available[: args.puzzles]
    entries = states_of(puzzles, len(puzzles))
    forward_puzzles = available[: max(args.puzzles, 96)]
    rows = build_questions(forward_puzzles, "benchmark", max(args.forward_sizes))
    training_rows = [
        row
        | dict(
            observed_outcome="yes" if index % 3 == 0 else "no",
        )
        for index, row in enumerate(build_questions(puzzles, "benchmark", 512))
    ]

    report = dict(
        config=dict(
            puzzles=len(puzzles),
            action_counts=[len(actions) for _, actions in entries],
            budget_gib=args.budget_gib,
        ),
        sections={},
        oom=[],
    )
    asked = list(args.forward_sizes)
    sizes = [size for size in asked if size <= len(rows)]
    if len(sizes) < len(asked):
        print(json.dumps(dict(skipped_sizes=[s for s in asked if s > len(rows)], available=len(rows))), flush=True)
    forward = sweep(
        "forward_fp32",
        sizes,
        lambda size: measure_forward(
            model,
            tokenizer,
            rows[:size],
            config,
            "fp32",
        ),
        args.budget_gib,
        report["oom"],
        report["sections"],
    )
    sweep(
        "forward_bf16",
        sizes,
        lambda size: measure_forward(model, tokenizer, rows[:size], config, "bf16"),
        args.budget_gib,
        report["oom"],
        report["sections"],
    )
    sweep(
        "rollout",
        args.rollout_states,
        lambda size: measure_rollout(model, tokenizer, entries, config, size),
        args.budget_gib,
        report["oom"],
        report["sections"],
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["head_lr"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    training = [
        training_rows[index % len(training_rows)]
        for index in range(args.effective_batch)
    ]
    train_rows = {}
    for objective in ("paired_pg", "brier"):
        train_rows[objective] = sweep(
            f"train_{objective}",
            args.train_sizes,
            lambda size, objective=objective: measure_training_step(
                model,
                tokenizer,
                training,
                config,
                optimizer,
                scheduler,
                size,
                objective,
            ),
            args.budget_gib,
            report["oom"],
            report["sections"],
        )
    report["recommendation"] = {}
    for name, inside in [("forward_fp32", forward)] + list(train_rows.items()):
        if not inside:
            report["recommendation"][name] = None
            continue
        best = max(inside, key=lambda row: row.get("questions_per_second", row.get("rows_per_second", 0.0)))
        largest = max(inside, key=lambda row: row["size"])
        report["recommendation"][name] = dict(
            throughput_choice=best, largest_inside_budget=largest
        )
    save_json(args.output, report)
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
