"""One logical full-parameter update; reused by online and legacy trainers."""

import time

import torch

from .dataset import collate
from .objectives import loss
from .serialization import OUTCOMES as OUTCOME_IDS


def optimization_step(model, tok, batch_rows, optimizer, scheduler, c, micro):
    device = next(model.parameters()).device
    step_start = time.monotonic()
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state() if device.type == "cuda" else None
    while True:
        try:
            model.train()
            optimizer.zero_grad(set_to_none=True)
            total = 0.0
            metrics = {}
            predicted_mass = torch.zeros(len(OUTCOME_IDS), device=device)
            observed_counts = torch.zeros_like(predicted_mass)
            for offset in range(0, len(batch_rows), micro):
                subset = batch_rows[offset : offset + micro]
                batch = collate(subset, tok, c["max_length"])
                with torch.autocast(
                    device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
                ):
                    lp = model(batch)
                targets = batch["targets"].to(device)
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
                        observed_nll=-lp.detach().gather(1, targets[:, None]).mean(),
                        observed_brier=(p.square().sum(1) - 2 * py + 1).mean(),
                        expected_reward_given_y=(2 * py - p.square().sum(1)).mean(),
                        target_probability=py.mean(),
                        entropy=-(p * lp.detach().masked_fill(~torch.isfinite(lp), 0))
                        .sum(1)
                        .mean(),
                        top1_accuracy=(p.argmax(1) == targets).float().mean(),
                        max_probability=p.max(1).values.mean(),
                    )
                    for key, value in diagnostics.items():
                        metrics[key] = metrics.get(key, 0) + value.detach() * len(
                            subset
                        ) / len(batch_rows)
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
            if cuda_rng is not None:
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
                metrics.pop("advantage_second_moment") - metrics["advantage_mean"] ** 2,
            )
            ** 0.5
        )
    step_seconds = time.monotonic() - step_start
    record = dict(
        objective=c["objective"],
        loss=total,
        gradient_norm=float(norm),
        gradient_clip_scale=min(1.0, 1.0 / (float(norm) + 1e-6)),
        microbatch=micro,
        gradient_accumulation=(len(batch_rows) + micro - 1) // micro,
        backbone_lr=learning_rates[0],
        head_lr=learning_rates[1],
        step_seconds=step_seconds,
        questions_per_second=len(batch_rows) / step_seconds,
        train=metrics,
        mean_predicted_distribution=dict(
            zip(OUTCOME_IDS, (predicted_mass / len(batch_rows)).tolist())
        ),
        observed_outcome_counts=dict(zip(OUTCOME_IDS, observed_counts.int().tolist())),
    )
    return record, micro
