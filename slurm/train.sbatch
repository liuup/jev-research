#!/usr/bin/env bash
# One full online RLCD run; the config controls generations and rollout sizes.
#SBATCH --job-name=jevmath-train
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=48G
#SBATCH --time=120:00:00
# Measured: one generation is about 3.2 h (collection dominates), so 20 generations
# plus the test evaluation need well over 48 h. The partition has no time limit.
#SBATCH --output=logs/slurm/%x-%j.out
set -euo pipefail
source slurm/startup.sh
RUN="${1:-runs/paired_pg}"
CONFIG="${2:-configs/train.yaml}"
uv run python scripts/train.py --config "$CONFIG" --output "$RUN"
uv run python scripts/evaluate.py --run "$RUN"
