# 24 点 Online RLCD

用 24 点游戏做「决策模型 + 在线 RLCD」实验：模型对每一步的每一个合法动作回答一个
yes/no 问题——**「现在执行这个动作，之后继续由冻结策略接管，最后的值会等于 24 吗？」**——
计算器精确执行每一步运算，标签来自真实 rollout 的观测结果。

## 1. 题库

枚举所有 4 个整数、取值 1–13、非递减排列的牌组，全部用 `Fraction` 精确计算：

| 项目 | 数量 |
| --- | --- |
| 全部牌组 | 1,820 |
| 可解 | 1,362 |
| 不可解 | 458 |

每道题都带元数据：

```json
{"puzzle_id": "3_3_8_8", "numbers": [3, 3, 8, 8], "target": 24, "solvable": true,
 "solution_count": 1, "example_solution": "8 / (3 - 8 / 3)"}
```

`solution_count` 统计**去重后的解**（交换律与相同数字造成的重复会合并），`example_solution`
取最短解；`3_3_8_8` 解唯一，正是 `8 / (3 - 8 / 3)`。

生成命令（约 7 秒，CPU 即可）：

```bash
uv run python scripts/build_24game.py --root data/24game --seed 17
```

产物与划分：

```text
data/24game/
├── train.jsonl          954 道可解题
├── dev.jsonl            204 道可解题
├── test.jsonl           204 道可解题
├── unsolvable.jsonl     458 道不可解题（稳健性集合，不进入训练）
├── oracle_actions.jsonl 每道可解题第一步的每个动作，以及执行后是否仍然可解
└── manifest.json        数字范围、规则、种子、规模、split 哈希、文件哈希、代码版本
```

防泄漏：分组键是排序后的数字多重集 `canonical_key = tuple(sorted(numbers))`。牌组在生成阶段
就已经按非递减形式保存，排列被合并，所以一题一组；`split_puzzles` 用固定种子确定性地按
954/204/204 切开，三个集合互不相交、并集正好是 1,362 道可解题。`verify_manifest` 会在每次
训练开始时校验行数与文件哈希，数据被改动就直接报错。

## 2. 环境

状态是**剩余数字的精确有理数多重集**（排序后的 `Fraction` 元组）。一步从两个数里选一对，
对有序数对施加 `+ - * /`：

- `+` 和 `*` 只提供一次（交换律），相同数值造成的重复动作按键 `op:left:right` 合并；
- `-` 和 `/` 两个方向都提供；
- 除数等于 0 的动作不生成；
- 事件结束条件只有「剩下一个数」，也就是固定三步运算；
- **没有 STOP 动作**；
- 最终值等于 24 记 1，否则记 0。

中间结果允许负数与分数，全程 `Fraction`，不做任何浮点近似。`solvable_after`（执行后是否
仍可解）只用于测试与 oracle 评估，不作为 RLCD 的概率标签。

## 3. 模型输入

一个 `(状态, 动作)` 对应两条候选路径，prompt 形如：

```text
[STATE]
Target: 24
Remaining: 8/3 3 8

Steps already taken:
1. 8 / 3 = 8/3

FrozenPolicy: 24game:seed17:solver0.5:g0

[ACTION]
3 - 8/3 = 1/3

[QUESTION]
If this action is performed now and the frozen policy continues, will the final value equal 24 after 2 more operations?

[OPTIONS]
YES
NO

[CANDIDATE]
YES
```

模型对每个动作输出 `P(YES)`，控制器取最大者（评估）或按归一化后的 `P(YES)` 采样
（采集）。prompt 里的 `FrozenPolicy` 标明这批标签是在哪个冻结策略下产生的，训练、dev、
test 三处使用同一个字符串，避免分布外输入。

## 4. 在线 RLCD 循环

每一代：

1. 冻结当前策略，给它一个 `policy_id`（`24game:seed17:g{generation}`）；
2. 从 train 里抽 `puzzles_per_generation` 道题，每题跑 `rollouts_per_puzzle` 次 rollout；
3. 每条 rollout 记录访问过的每个 `(state, action)`，**同一个 `(state, action)` 只保留一次**
   观测结果（本轮内去重，重复条数计入 `duplicate_rows`）；
4. 用 paired proper-reward PG 更新模型 `steps_per_generation` 步；
5. 训练前已经完成全部采集，learner 无法影响自己这一代的标签；
6. 代末在 dev 上做贪心评估并保存 checkpoint，最后在 test 与不可解题集上做一次终评。

**π0 的设定。** 全随机或完全未训练的策略成功率过低，会让标签几乎全是 `NO`。因此第 0 代用
`init_policy: solver_mix`：每一步以 `init_solver_mix` 的概率交给求解器（在「执行后仍可解」的
动作里等概率选一个），否则交给模型。第 1 代起完全由模型自己续玩，`FrozenPolicy` 也随之变成
`g1`。求解器只在第 0 代以这个方式参与数据生成，之后不再出现。

## 5. 实测性能与超参数

`uv run python scripts/benchmark_batches.py`（Slurm 作业，结果 `results/benchmark_batches.json`）
在 RTX 5090 上测于真实牌局状态（第一步动作数 10–22）：

| 形状 | 设置 | 峰值显存 | 吞吐 |
| --- | --- | --- | --- |
| 候选打分前向 bf16 | 16 / 64 / 512 题每次 | 3.9 / 4.7 / 11.8 GiB | 499 / 513 / 504 题/s |
| 候选打分前向 fp32 | 16 / 64 / 512 题每次 | 3.3 / 4.7 / 17.4 GiB | 195 / 213 / 209 题/s |
| 完整状态打分（真实收集路径） | 4 / 24 状态 | 3.94 GiB | 27.4 / 22.2 状态/s |
| 训练步 | microbatch 4 / 32 / 128 | 14.0 GiB | 42 / 110 / 116 行/s |

结论：

- **bf16 比 fp32 快 2.4 倍**，显存相同，因此 `inference_precision: bf16` 成为默认；
- 前向吞吐从 16 题起就进入平台（499→504 题/s），题数只影响显存，`inference_questions: 32`
  在平台上同时减少调用次数；
- 训练显存的地板是 14 GiB（fp32 参数 3 GB + AdamW 状态 6 GB + 梯度 3 GB + 激活），
  microbatch 从 4 加到 128 都不变，吞吐在 microbatch 16 之后到平台。

按上表，正式配置（`configs/train.yaml`）与预算：

| 阶段 | 量 | 耗时 |
| --- | --- | --- |
| 收集 | 954 题 × 4 次 rollout × 3 步 = 11,448 个状态 ≈ 23 万个候选问题 | 约 9 分钟 |
| 训练 | 300 步 × 64 行 = 19,200 次行前向 | 约 3 分钟 |
| dev 评估 + checkpoint | 204 题贪心 + 8.6 GB 写盘 | 约 1 分钟 |
| **每代合计** | | **约 13 分钟** |
| 30 代 + 终评 | | **约 7 小时** |

## 6. 运行

```bash
uv run pytest -q                                   # 49 个 CPU 测试
uv run python scripts/build_24game.py               # 生成题库与 manifest
uv run python scripts/smoke_test.py                 # CPU 全流程 smoke（tiny 模型）
sbatch slurm/build.sbatch                           # 需要 Slurm 的等价生成
sbatch slurm/smoke.sbatch                           # 10 步 GPU smoke（runs/smoke_gpu）
sbatch slurm/benchmark.sbatch                       # 批大小 / 显存探测
sbatch slurm/train.sbatch runs/paired_pg configs/train.yaml
sbatch slurm/train.sbatch runs/ce configs/ce.yaml
sbatch slurm/train.sbatch runs/brier configs/brier.yaml
uv run python scripts/evaluate.py --run runs/paired_pg   # 单独跑 test 评估
```

所有 GPU 工作必须在 Slurm 分配内运行（`runtime.require_slurm`）。

## 7. 产物与指标

```text
runs/<name>/
├── resolved_config.{json,yaml}  配置、manifest、题数、git commit
├── training.jsonl               每一步的 loss、梯度范数、校准诊断
├── generation_XXX/
│   ├── trajectories.jsonl       每条轨迹的动作、分布、熵、控制器
│   ├── rows.jsonl               去重后的训练行（一条一个观测结果）
│   ├── rollout_metrics.json     成功率、YES 比例、平均 P(YES)、动作熵、solver 占比
│   ├── dev_rows.jsonl / dev_metrics.json
├── generations.json / summary.json / checkpoint.pt（8.6 GB，每代覆盖）
└── test_metrics.json / unsolvable_metrics.json / test_report.json
```

指标口径：

- `solved_rate`、`first_action_solvable_rate`：贪心成功率，以及第一步是否保持可解（oracle）；
- `yes_rate`：这批行里观测结果为 YES 的比例；
- `binary_brier`、`observed_nll`、`ece_ge_correct`、`mean_prediction_entropy`：校准；
  `observed_brier` 是两类求和的多类 Brier，等于 `binary_brier` 的两倍，报表以 `binary_brier` 为准；
- `mean_action_entropy`：动作分布的熵（上限是 log 动作数），用来观察策略是否退化；
- `oracle_separation`：`P(YES)` 在「执行后仍可解」与「死路」两类动作上的均值差，直接衡量排序
  是否有效；
- `aurc` / risk-coverage：按置信度排序后的准确率曲线。

## 8. 已知限制

- 458 道不可解题在动作空间内确实无解，模型对它们唯一正确的行为是让 `P(YES)` 保持低位，
  因此它们只用于稳健性检查，不计入成功率。
- 第 0 代的续玩策略有一半由求解器完成，π0 的成功率因此偏高；从第 1 代起标签完全来自模型
  自身策略，跨代比较成功率时要注意这一点。
- 4 个数字固定三步运算，没有 STOP，所以「最终值」只在轨迹结束时存在，事件定义是
  「三步之后的值等于 24」，不存在中途报告答案的情形。
- dev 指标按行聚合，同一条轨迹的多行共享标签，有效样本量接近题数（204）而非行数。
