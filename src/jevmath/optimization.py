"""One logical full-parameter update shared by every objective."""

import time

import torch

from .dataset import collate
from .objectives import loss
from .serialization import candidates


def optimization_step(model, tokenizer, batch_rows, optimizer, scheduler, config, micro):
    device = next(model.parameters()).device
    candidate_ids = candidates(batch_rows[0])
    if any(candidates(row) != candidate_ids for row in batch_rows):
        raise ValueError("Training needs a consistent candidate order")
    started = time.monotonic()
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state() if device.type == "cuda" else None
    while True:
        try:
            model.train()
            optimizer.zero_grad(set_to_none=True)
            total = 0.0
            metrics = {}
            predicted_mass = torch.zeros(len(candidate_ids), device=device)
            observed_counts = torch.zeros_like(predicted_mass)
            for offset in range(0, len(batch_rows), micro):
                subset = batch_rows[offset : offset + micro]
                batch = collate(subset, tokenizer, config["max_length"])
                with torch.autocast(
                    device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
                ):
                    logp = model(batch)
                targets = batch["targets"].to(device)
                diagnostics = {}
                value = (
                    loss(
                        logp,
                        targets,
                        config["objective"],
                        config["reward_samples"],
                        config["baseline"],
                        diagnostics=diagnostics,
                    )
                    * len(subset)
                    / len(batch_rows)
                )
                value.backward()
                total += float(value.detach())
                with torch.no_grad():
                    probabilities = logp.detach().exp()
                    target_probability = probabilities.gather(
                        1, targets[:, None]
                    ).squeeze(1)
                    diagnostics.update(
                        observed_nll=-logp.detach()
                        .gather(1, targets[:, None])
                        .mean(),
                        observed_brier=(
                            probabilities.square().sum(1)
                            - 2 * target_probability
                            + 1
                        ).mean(),
                        target_probability=target_probability.mean(),
                        entropy=-(probabilities * logp.detach()).sum(1).mean(),
                        top1_accuracy=(probabilities.argmax(1) == targets).float().mean(),
                        predicted_success_probability=probabilities[:, 0].mean(),
                        observed_success_rate=(targets == 0).float().mean(),
                    )
                    for key, number in diagnostics.items():
                        metrics[key] = metrics.get(key, 0) + number.detach() * len(
                            subset
                        ) / len(batch_rows)
                    predicted_mass += probabilities.sum(0)
                    observed_counts += torch.bincount(targets, minlength=len(candidate_ids))
            break
        except torch.cuda.OutOfMemoryError:
            if micro == 1:
                raise
            optimizer.zero_grad(set_to_none=True)
            value = None
            batch = None
            micro = max(1, micro // 2)
            torch.cuda.empty_cache()
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state(cuda_rng)
            print(f"OOM: retrying logical batch with microbatch={micro}", flush=True)
    norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), 1.0, error_if_nonfinite=True
    )
    learning_rates = [group["lr"] for group in optimizer.param_groups]
    optimizer.step()
    scheduler.step()
    metrics = {key: float(number) for key, number in metrics.items()}
    if "advantage_second_moment" in metrics:
        metrics["advantage_std"] = (
            max(
                0,
                metrics.pop("advantage_second_moment") - metrics["advantage_mean"] ** 2,
            )
            ** 0.5
        )
    record = dict(
        objective=config["objective"],
        loss=total,
        gradient_norm=float(norm),
        microbatch=micro,
        gradient_accumulation=(len(batch_rows) + micro - 1) // micro,
        learning_rates=learning_rates,
        backbone_lr=learning_rates[0],
        head_lr=learning_rates[-1],
        step_seconds=time.monotonic() - started,
        train=metrics,
        mean_predicted_distribution=dict(
            zip(candidate_ids, (predicted_mass / len(batch_rows)).tolist())
        ),
        observed_outcome_counts=dict(
            zip(candidate_ids, observed_counts.int().tolist())
        ),
    )
    return record, micro
