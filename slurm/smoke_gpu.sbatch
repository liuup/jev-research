#!/usr/bin/env bash
#SBATCH --job-name=jev-smoke-seed17
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=48:00:00
#SBATCH --output=logs/slurm/%x-%j.out
set -euo pipefail
source slurm/startup.sh
uv run python scripts/smoke_test.py
