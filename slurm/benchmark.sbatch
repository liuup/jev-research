#!/usr/bin/env bash
# Batch size / memory probe for the 24 game workloads.
#SBATCH --job-name=24game-bench
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/%x-%j.out
set -euo pipefail
source slurm/startup.sh
uv run python scripts/benchmark_batches.py --output results/benchmark_batches.json
