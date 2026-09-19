#!/usr/bin/env bash
set -euo pipefail
mkdir -p logs/slurm
uv run pytest -q
uv run python scripts/audit_data.py --data-dir data/offline_pi0_v1
for objective in paired_pg ce brier; do
    job=$(sbatch --parsable --job-name="jev-offline-${objective}-seed17" slurm/train.sbatch "$objective" "$@")
    echo "$objective: $job"
done
