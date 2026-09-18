中文 | [English](README_en.md)

# 2048 game with online Jev (RLCD)

复现Jev, 通过环境反馈, 在2048游戏中使用RLCD进行进化.

默认使用本地 `/root/shang/hf-modles/Qwen3.5-0.8B-Base`，全参数训练文本骨干与候选集合评分头

## Online RLCD流程

1. 初始控制器为启发式策略。每代冻结当前控制器 `πt`。
2. 用 `πt + 15% 随机动作` 在线探索棋盘，对选中状态的每个合法动作独立分支；后续严格由 `πt` 玩到终局，取得一次观测 `Y`。
3. 更新 `p(y | s,a,πt)`；默认 paired-PG，32 个预测标签抽样，保留 detached 条件基线。也支持 CE/Brier。
4. 冻结新模型，构造贪心候选控制器。用独立同种子对局比较候选与当前策略，均分提高超过 2% 才晋升；否则保留当前策略，学习器继续训练。
5. 下一代重新采集反馈。权重延续、优化器重置.

无需预生成训练集。单卡采用同步“批量采样→更新”；环境在 CPU 执行，神经策略推理和训练都在 Slurm GPU 上执行

## 验证与启动

```bash
uv sync --locked
uv run pytest -q
uvx ruff check src scripts tests
uv run python scripts/validate_online.py
```

以下是之后的 GPU 验证/训练命令。先完成两阶段 smoke 并检查成功，再启动主实验：

```bash
mkdir -p logs/slurm
sbatch slurm/smoke_gpu.sbatch
# 上一作业通过后：
sbatch slurm/online_smoke.sbatch
# 在线 smoke 完成后：
uv run python scripts/verify_training_smoke.py --run runs/online_smoke

# 主实验（仅在准备好后手动提交）
sbatch --job-name=jev-online-pg-seed17 slurm/train.sbatch paired_pg
# 或三目标独立在线实验：
bash slurm/submit_all.sh
bash slurm/status.sh
# bash slurm/cancel.sh JOB_ID [JOB_ID ...]
```

配置位于 `configs/online.yaml`, 从本地 base 和 seed 初始化，也可指定 `initial_checkpoint`。三个目标初始权重与随机种子相同，但在线策略分化后采样轨迹会不同，比较时也需看环境交互量和耗时。

## 结果与评估

`runs/online_paired_pg_seed17/`：

- `training.jsonl`：reward、优势统计、NLL/Brier、熵、预测分布、采样/更新耗时、环境交互量及策略版本；不记录显存。
- `generation_XXX/events.jsonl`：本代实际用于更新的单次环境反馈，作为审计记录，不是预生成训练集。
- `generation_XXX/checkpoint.pt`：学习器权重；其预测目标由 `target_policy.json` 指定。显存统计另存 `training_stats.json`。
- `generation_XXX/metrics.json`：固定 holdout 棋盘上的概率指标、MC 动作排序一致率和效用 regret。每代按对应冻结策略重采标签与 MC 概率。
- `promotion.json`：晋升用对局；`controller_test.json`：独立测试对局，不参与晋升决定。
- `active_policy.json`：真正获准接管的策略，可能仍是启发式；不能把最后一个 checkpoint 自动当成更好的控制器。

```bash
sbatch slurm/evaluate.sbatch --run runs/online_paired_pg_seed17/generation_000
sbatch slurm/play.sbatch --run runs/online_paired_pg_seed17 --policy active --games 100 --seed 0
uv run python scripts/summarize.py
```

汇总输出为 `results/comparison.{csv,json,md}`，概率指标与闭环得分分表保存。训练、holdout、MC、晋升和测试使用隔离随机流；holdout 棋盘固定，MC 标签只用于评估。
