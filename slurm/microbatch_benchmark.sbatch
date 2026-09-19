#!/usr/bin/env bash
#SBATCH --job-name=jev-snake-microbatch-seed25
#SBATCH --partition=gpu1h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/slurm/%x-%j.out
set -euo pipefail
export OBJECTIVE=snake_microbatch_benchmark
source slurm/startup.sh
python scripts/benchmark_microbatch.py "$@"
