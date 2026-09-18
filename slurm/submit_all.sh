#!/usr/bin/env bash
set -euo pipefail
mkdir -p logs/slurm
uv run pytest -q
uv run python -c 'import json; assert json.load(open("results/gpu_smoke.json"))["success"]; assert json.load(open("results/training_smoke.json"))["success"]'
test -f data/main/mc_reference.jsonl
test -f runs/common_init.pt
for objective in ce brier paired_pg; do
    job=$(sbatch --parsable --job-name="jev-${objective}-seed17" slurm/train.sbatch "$objective" "$@")
    echo "$objective: $job"
    printf '%s\n' "$job" >> logs/submitted_jobs.txt
done
