#!/usr/bin/env bash
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
unset VIRTUAL_ENV
CONDA_ROOT=/mnt/apps/manual/miniforge3/25.3.1-0_gcc-11.4.1
set +u
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate pytorch
set -u
: "${LD_LIBRARY_PATH:=}"
export LD_LIBRARY_PATH=/mnt/apps/manual/miniforge3/25.3.1-0_gcc-11.4.1/lib:$LD_LIBRARY_PATH
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
hostname
date --iso-8601=seconds
git_ref=$(sed -n 's/^ref: //p' .git/HEAD)
if [[ -n "$git_ref" && -f ".git/$git_ref" ]]; then
    git_head=$(<".git/$git_ref")
else
    git_head=$(<.git/HEAD)
fi
echo "git_head=$git_head"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi
python --version
echo "conda_env=$CONDA_DEFAULT_ENV python=$(command -v python)"
echo "default_base_model=/root/shang/hf-modles/Qwen3.5-0.8B-Base default_seed=17 objective=${OBJECTIVE:-smoke}; resolved config follows in Python"
cat configs/model.yaml
