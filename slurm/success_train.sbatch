#!/usr/bin/env bash
#SBATCH --job-name=jev-success-pg-s17
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --cpus-per-task=24
#SBATCH --mem=0
#SBATCH --time=48:00:00
#SBATCH --output=logs/slurm/%x-%j.out
set -euo pipefail
export OBJECTIVE=paired_pg
source slurm/startup.sh
uv run python scripts/train.py \
    --config configs/success.yaml \
    --output runs/success_paired_pg_seed17
uv run python scripts/summarize.py \
    --runs runs/success_paired_pg_seed17 \
    --output results/success_comparison
