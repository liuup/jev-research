中文 | [English](README_en.md)

# Offline Jev / RLCD for 2048

本项目使用本地 `/root/shang/hf-modles/Qwen3.5-0.8B-Base` 研究静态数据上的 Jev 风格结果预测。模型不是动作分类器：对每个合法动作分别预测

`p(终局最大砖 | 当前棋盘, 首个动作, 冻结启发式 continuation π0)`。

输出为 `<256, 256, 512, 1024, 2048, 4096, 8192+` 上的完整概率分布。模型不生成文本；Qwen 文本骨干提取每个候选的表示，共享 set-attention 决策头在候选间归一化。视觉塔冻结，文本骨干和决策头全参数训练。

## 数据与防泄漏

固定行为策略从独立种子游戏采集棋盘。每个棋盘的所有合法首动作都从同一状态分支，随后固定启发式 `π0` 玩到终局。每个 `(state, action)` 仅保存一次真实随机终局事件，不保存 MC 概率作为训练标签。

`train/dev/calibration/test` 使用互不重叠的源游戏种子；同一源游戏和同一棋盘的所有动作不会跨 split。`manifest.json` 保存配置、策略/模拟器哈希、split 计数和文件哈希，`audit_data.py` 会再次验证隔离。MC reference 从 test 棋盘另行生成，只用于评估。

```bash
uv sync --locked
uv run pytest -q

# 通过 Slurm 生成约 100k/2k/2k/4k questions 并自动审计
sbatch slurm/generate_data.sbatch data/offline_pi0_v1
# 数据完成后构建 evaluation-only MC reference
sbatch slurm/build_mc_reference.sbatch data/offline_pi0_v1
```

启发式基准（100 局、seed 17）：mean score 15543.88，median 15084，达到 512/1024/2048 的概率为 0.96/0.72/0.18。训练数据生成后不得调整 `π0`。

## 训练目标

- `ce`：观测结果的负对数似然。
- `brier`：预测分布与观测 one-hot 的向量 Brier loss。
- `paired_pg`：32 个预测结果标签抽样和 detached 条件 baseline 的 RLCD-inspired proper-reward PG。这里的抽样是结果标签，不是 2048 动作。

三者使用完全相同的数据、batch schedule、seed、优化步数和 `runs/offline_common_init.pt`。主配置在 `configs/offline.yaml`，默认 500 steps、effective batch 256、microbatch 16。paired-PG 的环境反馈已经冻结在离线数据中；训练阶段不再 rollout。

```bash
# GPU 操作只能经 Slurm；先创建一次公共初始化
sbatch slurm/initialize.sbatch

# 单目标或三目标比较
sbatch --job-name=jev-offline-pg-seed17 slurm/train.sbatch paired_pg
bash slurm/submit_all.sh
bash slurm/status.sh
# bash slurm/cancel.sh JOB_ID [JOB_ID ...]
```

`training.jsonl` 记录 loss、梯度、reward/advantage、NLL、Brier、熵、top-1 和预测分布，不记录逐步耗时或显存。`training_stats.json` 单独保存总训练时间和峰值显存。测试报告在对应 `runs/offline_*_seed17/`；`scripts/summarize.py` 生成 `results/comparison.{json,csv,md}`。

NLL/Brier/ECE 衡量概率质量，top-1 不是校准。MC 概率从不参与训练。闭环游戏成绩是单独的控制演示，不能直接视为 `π0` 条件下的校准结论。
