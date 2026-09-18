#!/usr/bin/env bash
set -euo pipefail
if test -f logs/submitted_jobs.txt; then
    while IFS= read -r job; do scancel "$job"; done < logs/submitted_jobs.txt
fi
