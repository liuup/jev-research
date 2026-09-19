#!/usr/bin/env python3
"""Benchmark full training updates and recommend an H100 microbatch."""

import argparse
import json
import statistics
import time

import torch

from jevsnake.batching import collate
from jevsnake.collector import OnlineCollector
from jevsnake.config import load_config
from jevsnake.model import JevSnakeModel, load_tokenizer
from jevsnake.optimization import optimization_step
from jevsnake.policies import UniformPolicy
from jevsnake.runtime import require_slurm
from jevsnake.utils import save_json, seed_all


def optimizer_for(model, config):
    return torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": config["backbone_lr"]},
            {"params": model.head.parameters(), "lr": config["head_lr"]},
        ],
        weight_decay=config["weight_decay"],
        foreach=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/online.yaml")
    parser.add_argument(
        "--candidates", nargs="+", type=int, default=[8, 16, 32, 64, 128, 256]
    )
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument(
        "--output", default="results/snake_microbatch_benchmark_seed25.json"
    )
    parser.add_argument("--target-memory-fraction", type=float, default=0.85)
    args = parser.parse_args()
    require_slurm()
    config = load_config(args.config)
    seed_all(config["seed"])
    if sorted(set(args.candidates)) != args.candidates:
        raise ValueError("Candidates must be unique and ascending")
    if any(config["effective_batch"] % value for value in args.candidates):
        raise ValueError("Every candidate must divide effective_batch")
    if not 0 < args.target_memory_fraction < 1:
        raise ValueError("target-memory-fraction must be in (0,1)")
    if args.repeats < 1:
        raise ValueError("repeats must be positive")

    model = JevSnakeModel.load_base(config).float().cuda()
    tokenizer = load_tokenizer(config["base_model"])
    collector = OnlineCollector(UniformPolicy(), config, generation=0, stream="benchmark")
    rows = collector.collect(config["effective_batch"])
    sequence_length = collate(rows, tokenizer, config["max_length"])["tokens"][
        "input_ids"
    ].shape[1]
    optimizer = optimizer_for(model, config)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)

    # Allocate AdamW state before measuring candidate peaks.
    optimization_step(
        model,
        tokenizer,
        rows,
        optimizer,
        scheduler,
        config,
        args.candidates[0],
    )
    torch.cuda.synchronize()
    records = []
    for requested in args.candidates:
        elapsed_values = []
        peak_allocated = []
        peak_reserved = []
        actual_values = []
        for _ in range(args.repeats):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            started = time.monotonic()
            metrics, actual = optimization_step(
                model,
                tokenizer,
                rows,
                optimizer,
                scheduler,
                config,
                requested,
            )
            torch.cuda.synchronize()
            elapsed_values.append(time.monotonic() - started)
            peak_allocated.append(torch.cuda.max_memory_allocated() / 2**30)
            peak_reserved.append(torch.cuda.max_memory_reserved() / 2**30)
            actual_values.append(actual)
        free, total = torch.cuda.mem_get_info()
        elapsed = statistics.median(elapsed_values)
        row = {
            "requested_microbatch": requested,
            "actual_microbatches": actual_values,
            "fits_without_fallback": all(value == requested for value in actual_values),
            "elapsed_seconds": elapsed_values,
            "median_elapsed_seconds": elapsed,
            "events_per_second": len(rows) / elapsed,
            "peak_allocated_gib": max(peak_allocated),
            "peak_reserved_gib": max(peak_reserved),
            "free_after_step_gib": free / 2**30,
            "total_gib": total / 2**30,
            "loss": metrics["loss"],
        }
        records.append(row)
        print(json.dumps(row, allow_nan=False), flush=True)

    safe = [
        row
        for row in records
        if row["fits_without_fallback"]
        and row["peak_reserved_gib"]
        <= row["total_gib"] * args.target_memory_fraction
    ]
    if not safe:
        raise RuntimeError("No candidate fit inside the requested memory margin")
    recommendation = max(safe, key=lambda row: row["events_per_second"])
    result = {
        "seed": config["seed"],
        "gpu": torch.cuda.get_device_name(0),
        "objective": config["objective"],
        "effective_batch": config["effective_batch"],
        "event_horizon": config["event_horizon"],
        "sequence_length": sequence_length,
        "repeats": args.repeats,
        "target_memory_fraction": args.target_memory_fraction,
        "selection_rule": "highest median events_per_second below memory limit",
        "records": records,
        "recommended_microbatch": recommendation["requested_microbatch"],
    }
    save_json(args.output, result)
    print(json.dumps({"benchmark_complete": result}, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
