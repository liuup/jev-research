#!/usr/bin/env bash
# One generation with the tuned microbatch (32) and a slightly larger rollout set,
# to confirm the measured settings fit and run end to end.
#SBATCH --job-name=jevmath-smoke-tuned
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/%x-%j.out
set -euo pipefail
source slurm/startup.sh
uv run python scripts/train.py \
    --config configs/smoke.yaml \
    --output runs/smoke_tuned \
    --set problems_per_generation=8 \
    --set rollouts_per_problem=2 \
    --set max_steps=4 \
    --set steps_per_generation=4 \
    --set effective_batch=32 \
    --set microbatch=32 \
    --set validation_problems=4
