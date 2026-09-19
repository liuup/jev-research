# Online RLCD-inspired Snake

This repository trains a Qwen3.5-0.8B decision model from fresh online Snake
interaction.  The model does not emit an action directly.  For each legal
non-reverse action it predicts a complete distribution over three mutually
exclusive short-horizon events:

- `collision`: collision happens before the next food;
- `timeout`: neither collision nor food happens within the event horizon;
- `food`: food is collected before collision.

The controller uses the fixed utility `[-1, 0, +1]` and chooses the action with
the greatest predicted expected utility.  Training consumes only the single
event observed on the executed trajectory.  Evaluation-only Monte Carlo
references never enter the loss.

The sampled `paired_pg` objective is an independent RLCD-inspired proper-reward
estimator, not a reproduction of an undisclosed official Jev training recipe.
The deterministic Snake mechanics adapted from NanoJev are documented in
`THIRD_PARTY_NOTICES.md`.

## Layout

- `src/jevsnake/`: simulator, model, online collector, trainer and evaluation.
- `configs/`: H100 training and smoke-test settings.  Every seed is 25.
- `scripts/`: train, evaluate, play and validation entry points.
- `slurm/`: one-H100 Slurm launchers.
- `tests/`: deterministic simulator, online-event and objective checks.

Historical 2048 run directories and logs may remain on the cluster for
provenance, but no 2048 source code is part of this project.

## Validate on a CPU allocation

```bash
source ~/.bashrc
python -m unittest discover -s tests -v
python scripts/validate_config.py
```

## Train on Slurm

```bash
sbatch slurm/train.sbatch paired_pg
```

Use a fresh output directory for every run.  `ce`, `brier`, and `paired_pg`
remain first-class online controls.
