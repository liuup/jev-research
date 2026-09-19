#!/usr/bin/env bash
# Heuristic-judge sweep: the heuristic plans, the model judges only candidates whose
# heuristic value is within --judge-delta of the best move. Seeds 0..99 match the
# frozen-heuristic reference range.
# Usage: sbatch --job-name=jev-hybrid-d0 slurm/success_hybrid_sweep.sbatch 0
#SBATCH --job-name=jev-hybrid
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --output=logs/slurm/%x-%j.out
set -euo pipefail
export OBJECTIVE=closed_loop_hybrid_sweep
source slurm/startup.sh
DELTA="${1:?usage: sbatch slurm/success_hybrid_sweep.sbatch DELTA}"
uv run python scripts/play.py \
    --run runs/success_paired_pg_seed17 \
    --games 100 \
    --seed 0 \
    --batch-games 50 \
    --microbatch-questions 128 \
    --controller hybrid \
    --judge-delta "$DELTA"
