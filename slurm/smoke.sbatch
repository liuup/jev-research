#!/usr/bin/env bash
# Ten-step GPU smoke of the 24 game loop: one tiny generation, all metrics reported.
#SBATCH --job-name=24game-smoke
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/%x-%j.out
set -euo pipefail
source slurm/startup.sh
RUN="${1:-runs/smoke_gpu}"
rm -rf "$RUN"
uv run python scripts/train.py --config configs/smoke.yaml --output "$RUN"
