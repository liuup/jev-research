#!/usr/bin/env bash
set -euo pipefail
squeue -u "$(id -un)" -o '%.18i %.30j %.10T %.12M %.30R'
if (( $# > 0 )); then
    jobs=$(IFS=,; echo "$*")
    sacct -j "$jobs" --format=JobID,JobName%30,State,Elapsed,MaxRSS,ExitCode || true
fi
