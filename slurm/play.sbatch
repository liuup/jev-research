#!/usr/bin/env bash
#SBATCH --job-name=jev-demo-seed25
#SBATCH --partition=gpu1h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/slurm/%x-%j.out
set -euo pipefail
export OBJECTIVE=closed_loop_demo
source slurm/startup.sh
python scripts/play.py "$@"
python scripts/summarize.py
