#!/usr/bin/env bash
#SBATCH --job-name=jev-evaluate-seed25
#SBATCH --partition=gpu1h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/slurm/%x-%j.out
set -euo pipefail
export OBJECTIVE=evaluation
source slurm/startup.sh
python scripts/evaluate.py "$@"
