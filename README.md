# Jev-style 2048

用 Qwen3.5-0.8B 学习 `p(y | s,a,pi0)`：在棋盘 `s` 执行动作 `a`，随后使用冻结启发式策略 `pi0`，预测终局最大方块的概率分布。

候选结果为 `<256 / 256 / 512 / 1024 / 2048 / 4096 / 8192+`。架构为共享文本骨干＋候选集合注意力＋评分头，全参数微调，不使用视觉塔、LM 输出头或自回归生成。基础模型：`/root/shang/hf-modles/Qwen3.5-0.8B-Base`。

## 实验约定

- 训练只使用模拟器生成的单次观测结果 `Y`；MC 概率仅用于评估。按来源轨迹划分数据，生成后不得调整冻结策略。
- 比较 CE、直接 Brier 和 paired-PG。PG 从结果分布抽样 32 次，使用 detached 条件基线，期望奖励为 `2 pᵀq - ||p||²`。实现见 [objectives.py](src/jev2048/objectives.py)；它是 RLCD-inspired 方法，不是官方 TypeSafe RLCD 算法。
- 三种目标共享初始化、数据及优化设置。默认 500 步、microbatch 4、梯度累积 4、有效 batch 16，使用 BF16、梯度检查点和 AdamW。配置见 `configs/`，支持 `--set key=value`。
- 动作概率不等于结果概率，准确率不等于校准。闭环控制器表现单独报告。直接 Brier 可能因方差更低而优于 PG；单 seed 和有限 MC 样本不足以证明优越性。

## 安装与验证

使用 uv 管理环境；所有 GPU 工作通过 Slurm 提交，默认分区、单 GPU、48 小时限制。

```bash
uv sync --locked
uv run pytest -q
uvx ruff check src scripts tests
uv run python scripts/benchmark_heuristic.py --games 100
uv run python scripts/generate_data.py --output data/smoke --train-questions 32 --heldout-questions 8 --workers 4
mkdir -p logs/slurm
sbatch slurm/smoke_gpu.sbatch
```

CPU 测试包含精确 PG 梯度验证；GPU smoke 检查骨干和决策头的反向传播。验证失败时不要继续训练。

## 数据与训练

以下用于全新实验；已有数据、初始化和运行目录不会被覆盖。等待前一阶段完成后再继续。

```bash
# GPU smoke 通过后生成正式数据及评估专用 MC 参考集
uv run python scripts/generate_data.py --output data/main
uv run python scripts/build_mc_reference.py --pairs 256 --rollouts 256
uv run python scripts/audit_data.py

# 创建公共初始化，完成后先测试 paired-PG
sbatch --job-name=jev-init-seed17 slurm/train.sbatch paired_pg --initialize
sbatch --job-name=jev-pg-smoke-seed17 slurm/train.sbatch paired_pg \
  --output runs/paired_pg_smoke --set data_dir=data/smoke \
  --set steps=10 --set microbatch=1 --set eval_every=5 \
  --set eval_questions=8 --set warmup_steps=2

# smoke 完成后检查，再启动主实验
uv run python scripts/verify_training_smoke.py
sbatch --job-name=jev-paired_pg-seed17 slurm/train.sbatch paired_pg

# 完整三目标对照：用此命令替代上面的单目标提交
bash slurm/submit_all.sh
```

正式数据为 50,001 / 512 / 514 / 513 个 train/dev/calibration/test 问题。冻结策略的 100 局均分为 15,543.88，达到 512/1024/2048 的比例为 96%/72%/18%。训练集中没有 8192+，4096 也很少，稀有事件结论需谨慎。

## 日志、评估与演示

```bash
bash slurm/status.sh
tail -f logs/slurm/jev-paired_pg-mb4-seed17-29.out  # 当前主实验
# bash slurm/cancel.sh JOB_ID [JOB_ID ...]      # 显式指定要取消的作业

# 训练结束后；训练脚本也会自动执行最终概率评估
sbatch slurm/evaluate.sbatch --run runs/paired_pg_seed17
sbatch slurm/play.sbatch --run runs/paired_pg_seed17 --games 10 --utility threshold --threshold 2048
sbatch slurm/play.sbatch --run runs/paired_pg_seed17 --games 10 --utility log_tile
uv run python scripts/summarize.py
```

- `runs/<objective>_seed17/`：配置、逐步日志、checkpoint、预测和指标。
- `results/comparison.{csv,json,md}`：NLL、Brier、准确率、MC L2/MAE、ECE、显存和耗时；闭环表单独保存。
- `results/plots/`：可靠性图；ECE 使用 10 个固定等宽区间。

运行记录保存种子和数据/初始化哈希。GPU 运算不保证逐位确定性，当前不支持自动断点续训。闭环策略改变后，不能沿用冻结 `pi0` 下的校准解释。
