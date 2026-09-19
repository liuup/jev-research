#!/usr/bin/env bash
#SBATCH --job-name=jev-logtile-smoke-seed25
#SBATCH --partition=gpu1h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/slurm/%x-%j.out
set -euo pipefail
export OBJECTIVE=online_expected_log_tile_smoke
source slurm/startup.sh
python scripts/smoke_test.py
python scripts/train.py --config configs/online_smoke.yaml --output runs/online_expected_log_tile_smoke "$@"
python scripts/verify_training_smoke.py --run runs/online_expected_log_tile_smoke
