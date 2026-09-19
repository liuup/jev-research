#!/usr/bin/env bash
# Rebuild both comparison tables once every judge-delta run has finished.
#SBATCH --job-name=jev-hybrid-summary
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm/%x-%j.out
set -euo pipefail
export OBJECTIVE=closed_loop_hybrid_summary
source slurm/startup.sh
uv run python scripts/summarize.py \
    --runs runs/success_paired_pg_seed17 \
    --output results/success_comparison
