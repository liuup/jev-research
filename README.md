# Jev-style decision model for GSM8K reasoning steps

每一步不是生成整条思维链：模型在一个**结构化候选操作**集合上打分，输出
`P(最终答案正确 | s_t, a_t, pi)`。计算器执行具体运算，环境推进到下一个 reasoning state。

## 任务定义

- 初始题目池：GSM8K train prompts（官方 test 只用于最终评估）。
- reasoning state：当前已知量集合（题目中的数字 + 已经算出的中间量）与已执行步骤。
- 候选动作：`COMBINE:<op>:<i>:<j>`，`op ∈ {ADD, SUB, MUL, DIV}`；`ADD/MUL` 每个无序对一次，
  `SUB/DIV` 每个有序对一次，因此两种操作数顺序都可达；另有 `STOP:<i>` 表示把某个量作为最终答案。
  合并会消耗两个操作数并追加结果，所以从 `n` 个数字开始的 episode 最多 `n-1` 次合并。
  除零候选在构造阶段就被排除，所有值用 `Fraction` 精确计算。
- 事件：把最终答案与 GSM8K 的 `#### N` 比较，正确记 `Y=1`。
- episode 结束条件：模型选择 `STOP`、池子塌缩到只剩一个量、或达到 `max_steps`（此时按确定性规则
  取最后一个量）。

## 模型

`/root/shang/hf-modles/Qwen3.5-0.8B-Base` 的文本骨干，不使用自回归 LM head；新增 Jev 风格的
candidate scoring head（set attention + 掩码归一化），对每个 `(state, action)` 输出 yes/no 两路
概率。策略对动作的分布由各动作的 `P(yes)` 归一化得到（`rollout.action_distribution`）。

## 训练

每一代先冻结当前 policy，用它在线产生 trajectory（每题 `rollouts_per_problem` 次，
最多 `max_steps` 步），记录的 `(s_t, a_t)` 用该 trajectory 的最终结果作为观测事件，
再用 `paired_pg` / `ce` / `brier` 之一训练若干 optimizer steps，得到下一代 policy。
冻结策略按 generation 切换，避免 calibration target 每步漂移。

目标函数（`objectives.py`）：NanoJev 风格 proper-reward PG，`M=reward_samples` 次抽样，

```text
R    = (2/M) sum_i 1[A_i=Y] - sum_k c[k](c[k]-1)/(M(M-1)),   E[R] = 2 p.q - ||p||^2
b_i  = (2/M) p[Y] - 2 sum_{j!=i} p[A_j]/(M(M-1))
L_PG = -sum_i stop_gradient(r_i - b_i) log p[A_i],           E[grad] = grad ||p-q||^2
```

`tests/test_objectives.py` 用 float64 枚举验证了这条梯度恒等式。CE 与 Brier 作为对照臂保留，
用 `configs/ce.yaml`、`configs/brier.yaml` 在同样的数据与调度下运行。

## 数据划分

`data/gsm8k` 由 `scripts/fetch_gsm8k.py` 从官方 `grade-school-math` 仓库下载并记录 sha256。
train 里 95% 左右的问题（数字个数不超过 `max_quantities`）进入池子，再按 `validation_fraction`
切成 RL train 与 validation；官方 1,319 道 test 只在 `scripts/evaluate.py` 里使用，不参与任何 rollout。

## 运行

```bash
uv sync
uv run pytest -q
uv run python scripts/fetch_gsm8k.py      # CPU，需要网络
sbatch slurm/smoke.sbatch                  # 一次极小 generation，确认链路
sbatch slurm/train.sbatch runs/paired_pg configs/train.yaml
sbatch slurm/train.sbatch runs/ce configs/ce.yaml
sbatch slurm/train.sbatch runs/brier configs/brier.yaml
```

GPU 操作必须在 Slurm 分配内（`runtime.require_slurm`）。rollout 成本随
`problems_per_generation × rollouts_per_problem × max_steps × 每个状态的候选数` 增长，
候选数约为 `3m(m-1)+m`（`m` 为当前池子大小），先用 smoke 测出吞吐再放大。
`uv run python scripts/train.py --test --output runs/paired_pg` 在 test 上做最终评估。

## 评估

每个 generation 结束时在固定 validation 问题上用贪心策略跑一遍，报告：

- `final_answer_accuracy`：整题最终答案正确的比例。
- 概率质量：`observed_brier`（向量，等于标量二分类 Brier 的两倍）、`binary_brier`、`observed_nll`、
  `mean_entropy`、`ece_ge_correct` 与可靠性图（预测 0.8 的 state-action 实际成功率是否接近 80%）。
- `risk_coverage`：按置信度排序后的准确率曲线与 AURC。
- `action_ranking`（`action_reference_problems > 0` 时）：把模型对各动作的 `P(yes)` 排名与
  重复 rollout 得到的经验成功率排名比较，给出一致率与后悔值，这就是 step/action 层面的准确度。

产物：`runs/<name>/generations.json`、`training.jsonl`、`generation_XXX/{trajectories,rows,
validation_rows}.jsonl`、`generation_XXX/{rollout,validation}_metrics.json`、`checkpoint.pt`、
`summary.json`。

## 批大小、显存与吞吐（实测）

`uv run python scripts/benchmark_batches.py`（Slurm 作业，结果在 `results/benchmark_batches.json`）
在 RTX 5090 上测了三类形状：候选打分前向、完整状态打分（含分词）、训练步（含反向与优化器），
每项假设 26 GiB 的显存预算。模型本身 fp32 占 3 GB，实测峰值远低于卡容量。

| 形状 | 设置 | 峰值显存 | 吞吐 |
| --- | --- | --- | --- |
| 候选打分前向 | 16 题/次 | 1.85 GiB | 302 题/s（113k token/s） |
| 候选打分前向 | 64 题/次 | 3.83 GiB | 199 题/s（108k token/s） |
| 候选打分前向 | 512 题/次 | 20.52 GiB | 199 题/s（108k token/s） |
| 完整状态打分 | 16 状态 | 2.04 GiB | 3.3 状态/s（215 题/s） |
| 训练步（checkpointing 开） | microbatch 16 | 6.99 GiB | 52 行/s |
| 训练步（checkpointing 开） | microbatch 32 | 8.30 GiB | 52 行/s |
| 训练步（checkpointing 开） | microbatch 128 | 14.64 GiB | 47 行/s |
| 训练步（checkpointing 关） | microbatch 16 | 23.41 GiB | 63 行/s |
| 训练步（checkpointing 关） | microbatch 32 | 越界后回退到 8 | 60 行/s |

token 吞吐从 16 题往后就基本持平（107 到 116k token/s），所以把前向批次开大只增加显存、
不增加吞吐；`inference_questions: 16` 同时占优。fp32 与 bf16 autocast 的差别不到 2%，
精度不是显存杠杆。训练侧吞吐在 microbatch 16 到 32 之间达到平台（52 行/s），
关掉 gradient checkpointing 可以把训练提快 20%，但 microbatch 16 就要 23.4 GiB、
32 直接越界（`optimization_step` 会自动回退并记录实际的 microbatch），因此默认保持开启。

真正决定整体耗时的是收集而不是训练：状态打分约 3.3 状态/s，一代 12,000 条 trajectory
（每题约 3 步）就是 3.6 万个 state-step，约 3.3 小时；而 200 步训练只要约 4 分钟。
要缩短一代的时间，先调 `problems_per_generation`、`max_steps`、`max_quantities`，
而不是继续加大批次。

按上表把 `microbatch: 32`、`effective_batch: 64`、`inference_questions: 16` 写进
`configs/train.yaml` 后，`slurm/smoke_tuned.sbatch` 又跑了一次 8 题 × 2 次 rollout 的确认
generation：4 个训练步全部用 microbatch 32、accum 1，无回退，整轮峰值 13.15 GiB，
整个过程 29 秒（含模型加载）。显存不是这个任务的约束，32 GB 里有 18 GiB 以上空余。
