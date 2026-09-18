# Jev-style frozen-policy outcome prediction for 2048

This research prototype asks whether full-parameter Qwen3.5-0.8B can learn calibrated long-horizon event distributions from single stochastic observations, and compares cross entropy, direct vector Brier loss, and an RLCD-inspired paired proper-reward policy gradient. The paired method is **not the undisclosed official TypeSafe RLCD algorithm**. Direct Brier is a strong, lower-variance baseline; one seed cannot establish superiority.

For each board and legal first action, the model predicts `p(y | s,a,pi0)`, where `y` is the terminal maximum-tile bucket after taking that action and following the frozen heuristic `pi0` to termination. These are event probabilities, not probabilities of choosing LEFT/RIGHT/UP/DOWN. Accuracy alone does not establish calibration.

The outcome IDs are `lt256, 256, 512, 1024, 2048, 4096, 8192_plus`. Each question has one candidate path per outcome. The shared text backbone embeds deterministic plain text containing the board, score, action question, and candidate description. No chat template, generation, observed answer, or LM-head logits are used. The last non-padding hidden state is pooled; LayerNorm and a 256-dimensional projection feed one four-head candidate-set attention layer with a residual connection and shared scalar MLP. Masked softmax normalizes within each question. Candidate sets may differ in size and ordering; offsets, masks, candidate IDs, action IDs and state IDs are carried in the batch. All candidates across the question microbatch use one backbone forward.

The local checkpoint is `/root/shang/hf-modles/Qwen3.5-0.8B-Base`. Its safetensors text keys are loaded strictly into `Qwen3_5TextModel`. Vision parameters and the generative output head are excluded entirely. The tied input embedding remains trainable. The inspected Transformers implementation supports this text-only backbone ([upstream documentation](https://huggingface.co/docs/transformers/v5.3.0/model_doc/qwen3_5)).

## Environment and validation

```bash
uv sync --locked
uv run pytest -q
uv run python scripts/benchmark_heuristic.py --games 100 --workers 12
uv run python scripts/generate_data.py --output data/smoke --train-questions 32 --heldout-questions 8 --workers 4
mkdir -p logs/slurm
sbatch slurm/smoke_gpu.sbatch
```

The detected machine has one RTX 5090, 32 GB VRAM, NVIDIA driver 595.84 (CUDA capability 13.2), and Slurm's default `local` partition with `gpu:1`. Jobs use that default rather than hardcoding a partition. uv locks PyTorch 2.9.1 CUDA 13.0 and Transformers 5.3.0. Do not use pip, Conda, or activate an environment manually. Every GPU script requires `SLURM_JOB_ID`; all GPU jobs have a 48-hour limit.

CPU tests cover merge rules, invalid moves, cloning and seeds, masks, variable candidate counts, permutation equivariance, losses and reference generation. Exact float64 enumeration for `(K,M)=(2,2),(3,2),(2,3)` verifies the PG gradient equals the gradient of squared distance to `q`, with and without the detached conditional baseline. Do not proceed if these tests fail. The GPU smoke verifies all text and head gradients, finite normalized predictions, and absence of vision/output-head parameters, and records peak memory in `results/gpu_smoke.json`.

## Frozen policy and event data

`configs/heuristic.yaml` fixes the deterministic one-ply afterstate feature weights. Empty cells and merge opportunities are rewarded; monotonicity and smoothness use log2 tiles; the largest corner tile, maximum tile, and immediate merge score are rewarded. Ties use LEFT, RIGHT, UP, DOWN. The policy does not sample hypothetical tile spawns when choosing its action. The benchmark must precede dataset generation; policy config and source hashes are saved and checked. Do not tune the policy after generating labels.

Initial benchmark (seeds 0–99): mean score 15543.88, median 15084, maximum tiles `{256:4,512:24,1024:54,2048:18}`, reach probabilities 512=.96, 1024=.72, 2048=.18. Full provenance is in `results/heuristic_benchmark.json`.

The fixed behavior policy chooses a uniformly random legal action with probability .15 and otherwise uses `pi0`. Every twentieth state is selected. Every legal first action branches from the identical board/score/step state with an independently seeded simulator. One complete continuation yields exactly one categorical observed event. Separate seed ranges assign entire source games to train/dev/calibration/test; all actions of a selected state remain together, including at the size cutoff. Consequently requested question counts may be exceeded by up to three.

```bash
uv run python scripts/generate_data.py --output data/main
uv run python scripts/build_mc_reference.py --data-dir data/main --pairs 256 --rollouts 256 --workers 16
```

Generation settings are in `configs/data.yaml`; use `--train-questions`, `--heldout-questions`, and `--workers` to change the first dataset size. The manifest records resolved settings, source seeds, file hashes, and frozen policy hashes. CPU generation uses a process pool; its cost can be substantial because every label requires a terminal rollout.

MC reference pairs are sampled from the held-out test questions. Each has 256 independently seeded continuations and a stored count vector, probability estimate, and seed. `mc_reference.jsonl` and its manifest are separate evaluation-only artifacts; training explicitly rejects rows with `q`. MC estimates have sampling error and are not exact known probabilities; evaluation reports an estimated squared-L2 sampling floor. The calibration split is reserved and evaluated separately; no post-hoc calibration is fitted in this first experiment.

## Objectives and matched initialization

For predicted distribution `p` and observed label `Y`:

- CE: `-log p[Y]`.
- Vector Brier: `sum_k (p[k] - 1[k=Y])^2`.
- Paired PG: draw `M=32` outcome-label samples independently with replacement. With counts `c`, use `r_i=2/M*1[A_i=Y]-2*(c[A_i]-1)/(M*(M-1))` and detached baseline `b_i=2/M*p[Y]-2*sum_{j!=i}p[A_j]/(M*(M-1))`. Minimize `-sum_i stop_gradient(r_i-b_i)*log p[A_i]`, averaged over questions. `baseline: zero` is available for debugging.

The whole-sample reward is `R=2/M*sum_i 1[A_i=Y]-sum_k c[k]*(c[k]-1)/(M*(M-1))`. Its expectation is `2 p^T q-||p||² = ||q||²-||p-q||²`. Predictive draws are outcome labels, never physical game actions; `Y` is generated before training and independently of those draws. The conditional baseline depends on the other draws and is detached.

All objectives load the exact same saved initial weights, including the initialized head. No head-only warmup is currently used. Full text parameters plus the head are optimized with AdamW and BF16 autocast; FP32 parameter/optimizer storage avoids low-precision Adam updates. Training uses gradient checkpointing, `use_cache=False`, clipping at 1.0, linear warmup/decay, separate backbone/head LRs of 2e-5/2e-4, weight decay .01, seed 17, effective batch 16, microbatch 2, and max input length 512. Overlong inputs raise instead of silently truncating the candidate. A single logical batch is retried with a smaller microbatch on activation OOM. No LoRA, quantization, offload, or ZeRO is used.

```bash
# Create one common initialization through Slurm.
sbatch --job-name=jev-init-seed17 slurm/train.sbatch paired_pg --initialize

# Test paired PG FIRST, as requested. Inspect success before full experiments.
sbatch --job-name=jev-paired-pg-smoke-seed17 slurm/train.sbatch paired_pg \
  --output runs/paired_pg_smoke --set data_dir=data/smoke \
  --set steps=10 --set microbatch=1 --set effective_batch=16 \
  --set eval_every=5 --set eval_questions=8 --set warmup_steps=2

# After that job completes successfully, audit updates and record the gate.
uv run python scripts/verify_training_smoke.py

# Once smoke is verified and main data/MC are complete:
bash slurm/submit_all.sh
bash slurm/status.sh
bash slurm/cancel.sh
```

`submit_all.sh` checks CPU tests and recorded smoke success, then queues three objective jobs on the single GPU and prints/stores their IDs. Configuration is YAML plus repeated `--set key=value` overrides; the same Python implementation handles all objectives. To run only paired PG first on the real dataset:

```bash
sbatch --job-name=jev-paired-pg-seed17 slurm/train.sbatch paired_pg
```

The default comparison is 500 optimizer steps, dev evaluation every 50 steps, and the same seeded shuffled logical batch schedule for every objective. Predictive sampling does not change the batch order. Python/NumPy/PyTorch and simulator seeds are fixed. GPU kernels are not forced into fully deterministic mode; floating-point parallel reductions and backend kernels can introduce small differences. Transformers' portable PyTorch linear-attention path is used when optional fused libraries are absent.

Runs store the resolved config, git commit, manifest and initialization hashes, batch schedule hash, step logs, dev scores, final checkpoint with optimizer/scheduler and RNG states, held-out predictions, reliability data, elapsed training time, and allocated/reserved peak GPU memory. Training time includes periodic dev evaluation but excludes final checkpoint writing and final test evaluation. Checkpoint state is saved for reproducibility; automatic resumption is not currently implemented. Generated data, logs, plots and checkpoints are excluded from Git; `uv.lock` is committed.

## Probability evaluation and controller demonstration

```bash
sbatch slurm/evaluate.sbatch --run runs/paired_pg_seed17
uv run python scripts/summarize.py
sbatch slurm/play.sbatch --run runs/paired_pg_seed17 --games 10 --utility threshold --threshold 2048
sbatch slurm/play.sbatch --run runs/paired_pg_seed17 --games 10 --utility log_tile
uv run python scripts/summarize.py
```

Observed-event evaluation reports NLL, vector Brier, and top-1 accuracy. MC evaluation reports squared L2, per-coordinate mean absolute probability error, and natural-log Jensen–Shannon divergence. Derived events `>=1024`, `>=2048`, `>=4096` have binary Brier, ECE and reliability data using ten fixed bins `[0,.1),...,[.9,1]`. Reliability PNGs are saved under `results/plots/`. The summary writes JSON, CSV and Markdown probability tables, and a separate controller table; missing runs are never assigned fabricated metrics.

The controller re-queries all currently legal actions in a single batched evaluation after every actual stochastic game transition. It maximizes either `P(maximum >= threshold)` or expected log2 tile utility with bucket representatives `[7,8,9,10,11,12,13]`; the bottom and top buckets are approximations. Closed-loop scores, maximum-tile distribution and reach rates are demonstration results only. Probabilities learned under `pi0` must not be assumed calibrated under the new greedy controller.

The rollout interface accepts a continuation-policy object, leaving room for future frozen policy generations. Iterative improvement is not part of this first experiment. Main limitations are one training seed, finite MC accuracy, correlated questions within each held-out source trajectory, coarse outcome buckets, and compute-intensive closed-loop inference. Formal uncertainty comparisons should use trajectory-level resampling and additional training seeds.
