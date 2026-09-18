[中文](README.md) | English

# 2048 Game with Online Jev (RLCD)

This project reproduces the core idea of Jev and evolves a 2048 agent with RLCD using feedback from the environment.

By default, it uses the local `/root/shang/hf-modles/Qwen3.5-0.8B-Base` checkpoint and performs full-parameter training of the text backbone and candidate-set scoring head.

## Online RLCD Workflow

1. The initial controller is a heuristic policy. At generation `t`, the current controller `πt` is frozen.
2. Board states are explored online using `πt` mixed with 15% random actions. Every legal action is branched independently from each selected state, after which `πt` plays to termination to produce one observed outcome `Y`.
3. The model updates `p(y | s,a,πt)`. The default objective is paired-PG with 32 predictive-label samples and a detached conditional baseline. CE and Brier objectives are also supported.
4. The newly trained model is frozen and converted into a greedy candidate controller. The candidate and incumbent play independent games with matching seeds. The candidate is promoted only if its mean score exceeds the incumbent's by more than 2%; otherwise, the incumbent is retained and the learner continues training.
5. Feedback is collected again for the next generation. Model weights carry over, while the optimizer is reset.

No pre-generated training dataset is required. On a single GPU, the workflow alternates between batched sampling and updates. The environment runs on the CPU, while neural-policy inference and training run on a Slurm-managed GPU.

## Validation and Launch

```bash
uv sync --locked
uv run pytest -q
uvx ruff check src scripts tests
uv run python scripts/validate_online.py
```

The commands below run GPU validation and training. Complete both smoke-test stages and verify their success before launching the main experiment:

```bash
mkdir -p logs/slurm
sbatch slurm/smoke_gpu.sbatch
# After the previous job succeeds:
sbatch slurm/online_smoke.sbatch
# After the online smoke test completes:
uv run python scripts/verify_training_smoke.py --run runs/online_smoke

# Main experiment (submit manually only when ready)
sbatch --job-name=jev-online-pg-seed17 slurm/train.sbatch paired_pg
# Or launch three independent online objectives:
bash slurm/submit_all.sh
bash slurm/status.sh
# bash slurm/cancel.sh JOB_ID [JOB_ID ...]
```

Configuration is stored in `configs/online.yaml`. Training initializes from the local base checkpoint and configured seed, or optionally from `initial_checkpoint`. All three objectives begin with the same weights and random seed, but their sampling trajectories may diverge after their online policies diverge. Comparisons should therefore also account for environment interactions and elapsed time.

## Results and Evaluation

Under `runs/online_paired_pg_seed17/`:

- `training.jsonl`: reward, advantage statistics, NLL/Brier, entropy, predicted distributions, sampling/update timing, environment interaction counts, and policy versions; GPU memory is not recorded here.
- `generation_XXX/events.jsonl`: the single-outcome environment feedback actually consumed for updates in that generation. This is an audit log, not a pre-generated training dataset.
- `generation_XXX/checkpoint.pt`: learner weights. The corresponding prediction target is specified by `target_policy.json`. GPU-memory statistics are stored separately in `training_stats.json`.
- `generation_XXX/metrics.json`: probability metrics, Monte Carlo action-ranking agreement, and utility regret on fixed holdout boards. Labels and MC probabilities are regenerated each generation under that generation's frozen policy.
- `promotion.json`: games used for promotion decisions. `controller_test.json`: an independent test set that is not used for promotion.
- `active_policy.json`: the policy actually approved to control the game. It may still be the heuristic policy; the latest checkpoint must not automatically be treated as the stronger controller.

```bash
sbatch slurm/evaluate.sbatch --run runs/online_paired_pg_seed17/generation_000
sbatch slurm/play.sbatch --run runs/online_paired_pg_seed17 --policy active --games 100 --seed 0
uv run python scripts/summarize.py
```

Summary outputs are written to `results/comparison.{csv,json,md}`, with probability metrics and closed-loop scores kept in separate tables. Training, holdout, MC, promotion, and test data use isolated random streams. Holdout boards remain fixed, and MC labels are used for evaluation only.
