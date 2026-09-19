[中文](README.md) | English

# 2048 Game with Online Jev (RLCD)

This project reproduces the core idea of Jev and evolves a 2048 agent with RLCD using feedback from the environment.

By default, it uses the local `/work/s0liu022/hf-models/Qwen3.5-0.8B` checkpoint and performs full-parameter training of the text backbone and candidate-set scoring head.

## Online RLCD Workflow

1. The initial controller is a heuristic policy. At generation `t`, the current controller `πt` is frozen.
2. Board states are explored online using `πt` mixed with 15% random actions. Every legal action is branched independently from each selected state, after which `πt` plays to termination to produce one observed outcome `Y`.
3. The model updates `p(y | s,a,πt)`. The default objective is paired-PG with 32 predictive-label samples and a detached conditional baseline. CE and Brier objectives are also supported.
4. The newly trained model is frozen and chooses actions by expected terminal `log2(tile)`. Candidate and incumbent use matching game seeds; promotion requires the configured absolute improvement in mean terminal log-tile.
5. A generation advances only after promotion. Rejection keeps the current generation, model weights, Adam state, and learning-rate progress. Promotion freezes the new policy, resets the optimizer, and starts fresh feedback collection.

The input includes both the current board and the deterministic afterstate before random tile spawning, but never the terminal label. No pre-generated dataset is required. Heuristic continuation rollouts use asynchronous CPU prefetch; neural continuation and training use the Slurm-managed GPU.

The main budgets are in `configs/online.yaml`: `max_policy_generations` limits frozen policy versions, `min/max_optimizer_steps_per_policy` bound updates against one target policy, `promotion_interval_steps` controls promotion checks, and `max_total_optimizer_steps` is the global cap.

## Validation and Launch

```bash
source /mnt/apps/manual/miniforge3/25.3.1-0_gcc-11.4.1/etc/profile.d/conda.sh
conda activate pytorch
export LD_LIBRARY_PATH=/mnt/apps/manual/miniforge3/25.3.1-0_gcc-11.4.1/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python -m pytest -q
python -m ruff check src scripts tests
python scripts/validate_online.py
```

The cluster startup script activates the shared `pytorch` Conda environment. The Qwen 3.5 runtime is currently completed by PyTorch 2.10, Transformers 5.5, and Triton 3.6 from `~/.local`; the shared environment's PyTorch 2.5.1/Triton 3.1 stack is incompatible with this model. Install `pytest` and `ruff` before running tests and lint checks.

The commands below run GPU validation and training. Complete both smoke-test stages and verify their success before launching the main experiment:

```bash
mkdir -p logs/slurm
sbatch slurm/smoke_gpu.sbatch
# After the previous job succeeds:
sbatch slurm/online_smoke.sbatch
# After the online smoke test completes:
python scripts/verify_training_smoke.py --run runs/online_expected_log_tile_smoke

# Main experiment (submit manually only when ready)
sbatch --job-name=jev-online-pg-seed25 slurm/train.sbatch paired_pg
# Or launch three independent online objectives:
bash slurm/submit_all.sh
bash slurm/status.sh
# bash slurm/cancel.sh JOB_ID [JOB_ID ...]
```

Configuration is stored in `configs/online.yaml`. Training initializes from the local base checkpoint and configured seed, or optionally from `initial_checkpoint`. All three objectives begin with the same weights and random seed, but their sampling trajectories may diverge after their online policies diverge. Comparisons should therefore also account for environment interactions and elapsed time.

## Results and Evaluation

Under `runs/online_paired_pg_seed25/`:

- `training.jsonl`: reward, advantage statistics, NLL/Brier, expected-log-tile errors, predicted distributions, and environment interaction counts; GPU memory is not recorded here.
- `generation_XXX/events.jsonl`: the single-outcome environment feedback actually consumed for updates in that generation. This is an audit log, not a pre-generated training dataset.
- `generation_XXX/checkpoint.pt`: learner weights used by the latest promotion check. Its prediction target is specified by `target_policy.json`.
- `generation_XXX/metrics.json`: probability metrics, Monte Carlo action-ranking agreement, and utility regret on fixed holdout boards. Labels and MC probabilities are regenerated each generation under that generation's frozen policy.
- `promotion_step_XXXXXX.json`: each promotion check; `promotion.json` is the latest check. `controller_test.json` is independent and not used for promotion.
- `active_policy.json`: the policy actually approved to control the game. It may still be the heuristic policy; the latest checkpoint must not automatically be treated as the stronger controller.

```bash
sbatch slurm/evaluate.sbatch --run runs/online_paired_pg_seed25/generation_000
sbatch slurm/play.sbatch --run runs/online_paired_pg_seed25 --policy active --games 100 --seed 25
python scripts/summarize.py
```

Summary outputs are written to `results/comparison.{csv,json,md}`, with probability metrics and closed-loop scores kept in separate tables. Training, holdout, MC, promotion, and test data use isolated random streams. Holdout boards remain fixed, and MC labels are used for evaluation only.
