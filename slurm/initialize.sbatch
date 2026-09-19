#!/usr/bin/env bash
#SBATCH --job-name=jev-offline-init-seed17
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --cpus-per-task=24
#SBATCH --mem=0
#SBATCH --time=48:00:00
#SBATCH --output=logs/slurm/%x-%j.out
set -euo pipefail
export OBJECTIVE=common-initialization
source slurm/startup.sh
uv run python scripts/train.py --config configs/train_paired_pg.yaml --initialize

