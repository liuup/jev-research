#!/usr/bin/env bash
# One full 24 game RLCD run, then the test evaluation.
#SBATCH --job-name=24game-pg-s17
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --output=logs/slurm/%x-%j.out
set -euo pipefail
source slurm/startup.sh
RUN="${1:-runs/paired_pg}"
CONFIG="${2:-configs/train.yaml}"
uv run python scripts/train.py --config "$CONFIG" --output "$RUN"
uv run python scripts/evaluate.py --run "$RUN"
