#!/usr/bin/env bash
#SBATCH --job-name=jev-train-seed17
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --output=logs/slurm/%x-%j.out
set -euo pipefail
export OBJECTIVE="${1:-ce}"
source slurm/startup.sh
shift || true
uv run python scripts/train.py --config "configs/train_${OBJECTIVE}.yaml" "$@"
