#!/usr/bin/env bash
# Short end-to-end run for a first look: about 25 minutes at the measured throughput.
#SBATCH --job-name=jevmath-quick
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=logs/slurm/%x-%j.out
set -euo pipefail
source slurm/startup.sh
RUN="${1:-runs/quick}"
CONFIG="${2:-configs/quick.yaml}"
uv run python scripts/train.py --config "$CONFIG" --output "$RUN"
uv run python scripts/evaluate.py --run "$RUN"
