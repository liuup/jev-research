"""One full-model optimizer update with observable diagnostics."""

import time

import torch

from .batching import collate
from .events import OUTCOMES, UTILITIES
from .objectives import loss


def optimization_step(model, tokenizer, rows, optimizer, scheduler, config, microbatch):
    device = next(model.parameters()).device
    started = time.monotonic()
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state() if device.type == "cuda" else None
    while True:
        try:
            model.train()
            optimizer.zero_grad(set_to_none=True)
            total_loss = 0.0
            metrics = {}
            predicted_mass = torch.zeros(len(OUTCOMES), device=device)
            observed_counts = torch.zeros_like(predicted_mass)
            for offset in range(0, len(rows), microbatch):
                subset = rows[offset : offset + microbatch]
                batch = collate(subset, tokenizer, config["max_length"])
                with torch.autocast(
                    device.type,
                    dtype=torch.bfloat16,
                    enabled=device.type == "cuda",
                ):
                    logp = model(batch)
                targets = batch["targets"].to(device)
                diagnostics = {}
                objective = loss(
                    logp,
                    targets,
                    config["objective"],
                    config["reward_samples"],
                    config["baseline"],
                    diagnostics,
                ) * len(subset) / len(rows)
                objective.backward()
                total_loss += float(objective.detach())
                with torch.no_grad():
                    probabilities = logp.detach().exp()
                    target_probability = probabilities.gather(
                        1, targets[:, None]
                    ).squeeze(1)
                    utility = torch.tensor(
                        UTILITIES, dtype=probabilities.dtype, device=device
                    )
                    predicted_utility = probabilities @ utility
                    observed_utility = utility[targets]
                    diagnostics.update(
                        observed_nll=-logp.detach()
                        .gather(1, targets[:, None])
                        .mean(),
                        observed_brier=(
                            probabilities.square().sum(1) - 2 * target_probability + 1
                        ).mean(),
                        target_probability=target_probability.mean(),
                        entropy=-(
                            probabilities
                            * logp.detach().masked_fill(~torch.isfinite(logp), 0)
                        )
                        .sum(1)
                        .mean(),
                        top1_accuracy=(probabilities.argmax(1) == targets).float().mean(),
                        max_probability=probabilities.max(1).values.mean(),
                        utility_mae=(predicted_utility - observed_utility).abs().mean(),
                        utility_mse=(predicted_utility - observed_utility).square().mean(),
                    )
                    for key, value in diagnostics.items():
                        metrics[key] = metrics.get(key, 0) + value.detach() * len(
                            subset
                        ) / len(rows)
                    predicted_mass += probabilities.sum(0)
                    observed_counts += torch.bincount(
                        targets, minlength=len(OUTCOMES)
                    )
            break
        except torch.cuda.OutOfMemoryError:
            if microbatch == 1:
                raise
            optimizer.zero_grad(set_to_none=True)
            logp = objective = batch = None
            microbatch = max(1, microbatch // 2)
            torch.cuda.empty_cache()
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state(cuda_rng)
            print(f"OOM: retrying with microbatch={microbatch}", flush=True)
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), 1.0, error_if_nonfinite=True
    )
    learning_rates = [group["lr"] for group in optimizer.param_groups]
    optimizer.step()
    scheduler.step()
    metrics = {key: float(value) for key, value in metrics.items()}
    if "advantage_second_moment" in metrics:
        metrics["advantage_std"] = max(
            0.0,
            metrics.pop("advantage_second_moment") - metrics["advantage_mean"] ** 2,
        ) ** 0.5
    elapsed = time.monotonic() - started
    return (
        {
            "objective": config["objective"],
            "loss": total_loss,
            "gradient_norm": float(gradient_norm),
            "gradient_clip_scale": min(1.0, 1.0 / (float(gradient_norm) + 1e-6)),
            "microbatch": microbatch,
            "gradient_accumulation": (len(rows) + microbatch - 1) // microbatch,
            "backbone_lr": learning_rates[0],
            "head_lr": learning_rates[1],
            "step_seconds": elapsed,
            "events_per_second": len(rows) / elapsed,
            "train": metrics,
            "mean_predicted_distribution": dict(
                zip(OUTCOMES, (predicted_mass / len(rows)).tolist())
            ),
            "observed_outcome_counts": dict(
                zip(OUTCOMES, observed_counts.int().tolist())
            ),
        },
        microbatch,
    )
