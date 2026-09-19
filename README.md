中文 | [English](README_en.md)

# 2048 game with online Jev (RLCD)

复现Jev, 通过环境反馈, 在2048游戏中使用RLCD进行进化.

默认使用本地 `/work/s0liu022/hf-models/Qwen3.5-0.8B`，全参数训练文本骨干与候选集合评分头

## Online RLCD流程

1. 初始控制器为启发式策略。每代冻结当前控制器 `πt`。
2. 用 `πt + 15% 随机动作` 在线探索棋盘，对选中状态的每个合法动作独立分支；后续严格由 `πt` 玩到终局，取得一次观测 `Y`。
3. 更新 `p(y | s,a,πt)`；默认 paired-PG，32 个预测标签抽样，保留 detached 条件基线。也支持 CE/Brier。
4. 冻结新模型，按终局桶的 `log2(tile)` 期望选择动作。候选与当前策略使用相同种子对局；平均终局 log-tile 达到配置的绝对提升才晋升。
5. 只有策略晋升才增加 generation。晋升失败时继续当前 generation，并保留模型权重、Adam 状态和学习率进度；晋升后冻结新策略、重置优化器并重新采集反馈。

输入同时包含原棋盘和执行候选动作后的确定性 afterstate（随机生成新砖之前），但不包含终局标签。无需预生成训练集；启发式 continuation 使用 CPU 异步预取，神经 continuation 和训练均在 Slurm GPU 上执行。

核心预算位于 `configs/online.yaml`：`max_policy_generations` 限制冻结策略版本数，`min/max_optimizer_steps_per_policy` 限制同一目标策略下的更新数，`promotion_interval_steps` 控制晋升检查频率，`max_total_optimizer_steps` 提供全局上限。

## 验证与启动

```bash
source /mnt/apps/manual/miniforge3/25.3.1-0_gcc-11.4.1/etc/profile.d/conda.sh
conda activate pytorch
export LD_LIBRARY_PATH=/mnt/apps/manual/miniforge3/25.3.1-0_gcc-11.4.1/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python -m pytest -q
python -m ruff check src scripts tests
python scripts/validate_online.py
```

集群启动脚本会激活共享 `pytorch` Conda 环境。Qwen 3.5 运行栈当前由 `~/.local` 中的 PyTorch 2.10、Transformers 5.5 和 Triton 3.6 补齐；共享环境自带的 PyTorch 2.5.1/Triton 3.1 与该模型不兼容。运行测试和代码检查前还需分别安装 `pytest` 与 `ruff`。

以下是之后的 GPU 验证/训练命令。先完成两阶段 smoke 并检查成功，再启动主实验：

```bash
mkdir -p logs/slurm
sbatch slurm/smoke_gpu.sbatch
# 上一作业通过后：
sbatch slurm/online_smoke.sbatch
# 在线 smoke 完成后：
python scripts/verify_training_smoke.py --run runs/online_expected_log_tile_smoke

# 主实验（仅在准备好后手动提交）
sbatch --job-name=jev-online-pg-seed25 slurm/train.sbatch paired_pg
# 或三目标独立在线实验：
bash slurm/submit_all.sh
bash slurm/status.sh
# bash slurm/cancel.sh JOB_ID [JOB_ID ...]
```

配置位于 `configs/online.yaml`, 从本地 base 和 seed 初始化，也可指定 `initial_checkpoint`。三个目标初始权重与随机种子相同，但在线策略分化后采样轨迹会不同，比较时也需看环境交互量和耗时。

## 结果与评估

`runs/online_paired_pg_seed25/`：

- `training.jsonl`：reward、优势统计、NLL/Brier、expected log-tile 误差、预测分布及环境交互量；不记录显存。
- `generation_XXX/events.jsonl`：本代实际用于更新的单次环境反馈，作为审计记录，不是预生成训练集。
- `generation_XXX/checkpoint.pt`：最后一次晋升检查使用的学习器权重；其预测目标由 `target_policy.json` 指定。
- `generation_XXX/metrics.json`：固定 holdout 棋盘上的概率指标、MC 动作排序一致率和效用 regret。每代按对应冻结策略重采标签与 MC 概率。
- `promotion_step_XXXXXX.json`：各次晋升检查；`promotion.json` 是最后一次检查；`controller_test.json` 是不参与晋升的独立测试。
- `active_policy.json`：真正获准接管的策略，可能仍是启发式；不能把最后一个 checkpoint 自动当成更好的控制器。

```bash
sbatch slurm/evaluate.sbatch --run runs/online_paired_pg_seed25/generation_000
sbatch slurm/play.sbatch --run runs/online_paired_pg_seed25 --policy active --games 100 --seed 25
python scripts/summarize.py
```

汇总输出为 `results/comparison.{csv,json,md}`，概率指标与闭环得分分表保存。训练、holdout、MC、晋升和测试使用隔离随机流；holdout 棋盘固定，MC 标签只用于评估。
