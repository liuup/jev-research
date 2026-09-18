#!/usr/bin/env bash
#SBATCH --job-name=jev-online-smoke-seed17
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --output=logs/slurm/%x-%j.out
set -euo pipefail
export OBJECTIVE=online_paired_pg_smoke
source slurm/startup.sh
uv run python scripts/train.py --config configs/online_smoke.yaml --output runs/online_smoke "$@"
