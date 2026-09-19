#!/usr/bin/env bash
# Closed-loop control of the trained reach_2048 model with several games sharing
# one batched forward. Seeds 0..99 match the frozen-heuristic reference range.
#SBATCH --job-name=jev-success-play-batch-s17
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --output=logs/slurm/%x-%j.out
set -euo pipefail
export OBJECTIVE=closed_loop_play_batched
source slurm/startup.sh
uv run python scripts/play.py \
    --run runs/success_paired_pg_seed17 \
    --games 100 \
    --seed 0 \
    --batch-games 50 \
    --microbatch-questions 128
uv run python scripts/benchmark_heuristic.py --games 100 --workers 4
uv run python scripts/summarize.py \
    --runs runs/success_paired_pg_seed17 \
    --output results/success_comparison
