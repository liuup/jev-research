#!/usr/bin/env bash
# One tiny generation end to end: dataset, rollouts, paired-PG update, evaluation.
#SBATCH --job-name=jevmath-smoke
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/%x-%j.out
set -euo pipefail
source slurm/startup.sh
[ -f data/gsm8k/train.jsonl ] || uv run python scripts/fetch_gsm8k.py
uv run python scripts/smoke_test.py
