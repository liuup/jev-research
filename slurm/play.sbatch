#!/usr/bin/env bash
#SBATCH --job-name=jev-demo-seed9017
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=48:00:00
#SBATCH --output=logs/slurm/%x-%j.out
set -euo pipefail
export OBJECTIVE=closed_loop_demo
source slurm/startup.sh
uv run python scripts/play.py "$@"
