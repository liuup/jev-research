#!/usr/bin/env bash
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
unset VIRTUAL_ENV
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
hostname
date --iso-8601=seconds
git rev-parse HEAD
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi
uv run python --version
uv --version
echo "base_model=/root/shang/hf-modles/Qwen3.5-0.8B-Base seed=17 objective=${OBJECTIVE:-smoke}"
cat configs/model.yaml
