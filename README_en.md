[中文](README.md) | English

# Offline Jev / RLCD for 2048

This project uses the local `/root/shang/hf-modles/Qwen3.5-0.8B-Base` checkpoint to study Jev-style outcome prediction on a static dataset. It is not an action classifier. For every legal action it separately predicts

`p(terminal maximum tile | board, first action, frozen heuristic continuation π0)`.

The output is a complete distribution over `<256, 256, 512, 1024, 2048, 4096, 8192+`. No text is generated. The Qwen text backbone encodes each candidate and a shared set-attention decision head normalizes across candidates. The vision tower is frozen; the text backbone and decision head are fully fine-tuned.

## Data and leakage prevention

A fixed behavior policy collects boards from independently seeded games. Every legal first action is branched from the same board, after which the frozen heuristic `π0` plays to termination. Each `(state, action)` stores exactly one observed stochastic terminal event; MC probabilities are never training labels.

`train/dev/calibration/test` use disjoint source-game seed ranges. All actions from a board and all states from a source game stay in one split. `manifest.json` stores settings, policy/simulator hashes, split counts, and file hashes; `audit_data.py` verifies the isolation. A separate MC reference is sampled from test boards for evaluation only.

```bash
uv sync --locked
uv run pytest -q

# Generate about 100k/2k/2k/4k questions through Slurm and audit them
sbatch slurm/generate_data.sbatch data/offline_pi0_v1
# After data generation, build the evaluation-only MC reference
sbatch slurm/build_mc_reference.sbatch data/offline_pi0_v1
```

Heuristic benchmark (100 games, seed 17): mean score 15543.88, median 15084, with reach probabilities 0.96/0.72/0.18 for 512/1024/2048. `π0` must not change after data generation.

## Training objectives

- `ce`: negative log likelihood of the observed outcome.
- `brier`: vector Brier loss against the observed one-hot event.
- `paired_pg`: RLCD-inspired proper-reward PG with 32 predictive outcome-label samples and a detached conditional baseline. These samples are outcome labels, not 2048 actions.

All objectives use identical data, batch schedule, seed, update count, and `runs/offline_common_init.pt`. The main settings are in `configs/offline.yaml`: 500 steps, effective batch 256, and microbatch 16. Environment feedback for paired-PG is already frozen in the offline dataset; training performs no rollouts.

```bash
# GPU work must use Slurm; create the common initialization once
sbatch slurm/initialize.sbatch

# One objective or the matched three-objective comparison
sbatch --job-name=jev-offline-pg-seed17 slurm/train.sbatch paired_pg
bash slurm/submit_all.sh
bash slurm/status.sh
# bash slurm/cancel.sh JOB_ID [JOB_ID ...]
```

`training.jsonl` records loss, gradients, reward/advantage diagnostics, NLL, Brier, entropy, top-1, and predicted distributions, but not per-step timing or GPU memory. `training_stats.json` separately stores total training time and peak memory. Test reports are written under `runs/offline_*_seed17/`; `scripts/summarize.py` creates `results/comparison.{json,csv,md}`.

NLL, Brier, and ECE measure probability quality; top-1 is not calibration. MC probabilities never enter training. Closed-loop game scores are a separate control demonstration and are not calibration evidence under `π0`.
