#!/usr/bin/env bash
#SBATCH --job-name=jev-success-pg-smoke-s17
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --output=logs/slurm/%x-%j.out
set -euo pipefail
export OBJECTIVE=paired_pg
source slurm/startup.sh
if [ ! -f runs/success_common_init.pt ]; then
    uv run python scripts/train.py --config configs/success_smoke.yaml --initialize
fi
uv run python scripts/train.py --config configs/success_smoke.yaml --output runs/success_paired_pg_smoke_seed17
