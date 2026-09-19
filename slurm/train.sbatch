#!/usr/bin/env bash
#SBATCH --job-name=jev-offline-train-seed17
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --cpus-per-task=24
#SBATCH --mem=0
#SBATCH --time=48:00:00
#SBATCH --output=logs/slurm/%x-%j.out
set -euo pipefail
export OBJECTIVE="${1:-paired_pg}"
source slurm/startup.sh
shift || true
uv run python scripts/train.py --config "configs/train_${OBJECTIVE}.yaml" "$@"
uv run python scripts/summarize.py --runs "runs/offline_${OBJECTIVE}_seed17"
