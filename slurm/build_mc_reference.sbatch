#!/usr/bin/env bash
#SBATCH --job-name=jev-offline-mc
#SBATCH --exclusive
#SBATCH --cpus-per-task=24
#SBATCH --mem=0
#SBATCH --time=48:00:00
#SBATCH --output=logs/slurm/%x-%j.out
set -euo pipefail
export OBJECTIVE=offline-mc-reference
source slurm/startup.sh
data_dir="${1:-data/offline_pi0_v1}"
shift || true
uv run python scripts/build_mc_reference.py --data-dir "$data_dir" "$@"

