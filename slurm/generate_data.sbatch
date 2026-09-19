#!/usr/bin/env bash
#SBATCH --job-name=jev-offline-data
#SBATCH --exclusive
#SBATCH --cpus-per-task=24
#SBATCH --mem=0
#SBATCH --time=48:00:00
#SBATCH --output=logs/slurm/%x-%j.out
set -euo pipefail
export OBJECTIVE=offline-data
source slurm/startup.sh
output="${1:-data/offline_pi0_v1}"
shift || true
uv run python scripts/generate_data.py --output "$output" "$@"
uv run python scripts/audit_data.py --data-dir "$output"

