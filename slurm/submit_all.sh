#!/usr/bin/env bash
set -euo pipefail
mkdir -p logs/slurm
uv run pytest -q
uv run python scripts/validate_online.py
uv run python scripts/verify_training_smoke.py --run runs/online_smoke
for objective in paired_pg ce brier; do
    job=$(sbatch --parsable --job-name="jev-online-${objective}-seed17" slurm/train.sbatch "$objective" "$@")
    echo "$objective: $job"
done
