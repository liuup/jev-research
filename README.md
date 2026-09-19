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
