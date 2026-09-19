#!/usr/bin/env bash
# Probe batch sizes against GPU memory: rollout forwards, then training steps.
#SBATCH --job-name=jevmath-bench
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/%x-%j.out
set -euo pipefail
source slurm/startup.sh
uv run python scripts/benchmark_batches.py "$@"
