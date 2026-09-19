#!/usr/bin/env bash
set -euo pipefail
CONDA_ROOT=/mnt/apps/manual/miniforge3/25.3.1-0_gcc-11.4.1
set +u
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate pytorch
set -u
: "${LD_LIBRARY_PATH:=}"
export LD_LIBRARY_PATH=/mnt/apps/manual/miniforge3/25.3.1-0_gcc-11.4.1/lib:$LD_LIBRARY_PATH
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p logs/slurm
python -m pytest -q
python scripts/validate_online.py
python scripts/verify_training_smoke.py --run runs/online_smoke
for objective in paired_pg ce brier; do
    job=$(sbatch --parsable --job-name="jev-online-${objective}-seed25" slurm/train.sbatch "$objective" "$@")
    echo "$objective: $job"
done
