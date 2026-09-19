#!/usr/bin/env bash
# One full online RLCD run; the config controls generations and rollout sizes.
#SBATCH --job-name=jevmath-train
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/slurm/%x-%j.out
set -euo pipefail
source slurm/startup.sh
RUN="${1:-runs/paired_pg}"
CONFIG="${2:-configs/train.yaml}"
uv run python scripts/train.py --config "$CONFIG" --output "$RUN"
uv run python scripts/evaluate.py --run "$RUN"
