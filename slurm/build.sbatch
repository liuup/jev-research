#!/usr/bin/env bash
# Generate data/24game: puzzle files, oracle actions and the manifest. No GPU needed.
#SBATCH --job-name=24game-build
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/%x-%j.out
set -euo pipefail
source slurm/startup.sh
uv run python scripts/build_24game.py --root data/24game --seed 17
