#!/usr/bin/env bash
set -euo pipefail
if (( $# == 0 )); then
    echo "Usage: bash slurm/cancel.sh JOB_ID [JOB_ID ...]" >&2
    exit 2
fi
for job in "$@"; do
    [[ "$job" =~ ^[0-9]+$ ]] || { echo "Invalid job ID: $job" >&2; exit 2; }
done
scancel "$@"
