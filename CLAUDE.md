# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# 赛道一：储层参数（POR / PERM / SW）测井智能预测

## 0. 任务与验收标准（唯一目标，先读这一节）

**交付两个 zip，缺一不可：**

| 工件 | 规范 |
|---|---|
| `result.zip` | 内含单个 `result.json`，10 口测试井逐深度点预测，schema 不得增删改任何字段 |
| `submission_code.zip` | 严格按下方 §3.2 的 11 个文件树，可一键复现：`python predict.py --data_dir ./data --output ./result.json` |

**硬性分数线（官方，来自 `参数说明.docx`）：本赛题基线 75 分，B 榜综合得分必须超过该线才能晋级决赛。**

当前状态与 75 分的差距是本项目的核心事实，见 §4。所有优化工作的意义都由这条线定义。

工作协议（每一轮优化都按此走）：
1. **复核评分标准** —— 任何"分数"主张必须先确认用的是官方口径（§2），不是自造变体。
2. **测试** —— 用固定切分评估（§6），必须带分解表（各目标 acc / acc_def / acc_real）。
3. **优化** —— 只采纳通过晋级规则（§6）的改动；跨工件比较必须固定同一套融合权重与阈值。
4. **整理** —— 用 `tools/package_submission.py` 生成工件，不手工拼 zip。

## 1. 数据事实（决定建模策略）

- 训练 80 井 / ~730K 深度点（`train/`），测试 10 井 / ~96K 点（`test/`），采样间隔 0.1 m。
- 每井一个 `.txt`；前两行是表头 + 单位行，数据从第 3 行起。训练 17 列（含 POR/PERM/SW），测试 14 列。
- `logId` = 文件名去掉 `.txt`。
- **CASE 列缺失率 ~99%，剔除。** 特征集 = 13 条原始曲线（含 DEPTH，不含 CASE）。
- 哨兵值 `-99999`、`-9999`、`-2.534573822542048e37` 一律视为缺失；CNL 物理范围 [0,100]%，越界截断。
- **核心结构**：存在"非储层默认段"（三目标同时退化为常数 POR=0.1 / PERM=0.01 / SW=99.9）与"储层真实段"。默认段预测准确即满分。真实段沿深度高度平滑（POR lag-1 自相关 0.967）。

## 2. 评分标准（官方口径，不要漂移）

ε = 1e-5，逐深度点独立、等权：

```
Acc_POR  = mean( max(0, 1 − |ŷ − y| / (0.08 · (|y| + ε))) )
Acc_PERM = mean( max(0, 1 − |log10(max(ŷ,ε)) − log10(max(y,ε))| / (0.08 · (|log10(max(y,ε))| + ε))) )
Acc_SW   = mean( max(0, 1 − |ŷ − y| / (0.05 · (|y| + ε))) )
Total    = (0.30·Acc_POR + 0.35·Acc_PERM + 0.35·Acc_SW) × 100
```

- POR / SW 在普通空间，**PERM 在 log10 空间**；容差 POR 8%、PERM 8%、SW 5%（相对）。
- 实现：`src/utils.py` 的 `acc_single` / `total_score`（`solution/metric.py` 是同一份的旧拷贝）。
- 仓库内所有评估脚本共用这一份实现。**不要在脚本里就地改公式**；需要新变体时单独建脚本并注明。

## 3. 提交规范

### 3.1 `result.json` schema（禁止增删改字段）

```json
{
  "modelId": "",
  "modelName": "模型名称",
  "version": "5.0",
  "resultData": [
    {
      "logId": "d122f5424b9e4984a8f701d16c2d5b58",
      "predictions": [
        { "depth": 1117.1, "POR": 21.7214, "PERM": 16.624, "SW": 79.0923 }
      ]
    }
  ]
}
```

`predictions` 按 depth 升序；每个点必须且只能是 `{depth, POR, PERM, SW}`；数值必须有限（无 NaN/Inf）。

### 3.2 `submission_code.zip` 文件树（11 个规范文件，顶层不得有多余项）

```
README.md  predict.py  requirements.txt  config.yaml  train.py
src/model.py  src/feature.py  src/utils.py
models/model.pkl  models/scaler.pkl
examples/result_example.json
```

（`src/__init__.py` 由打包脚本自动补齐，不算规范文件。）
README.md 必须含：算法名称、方法简介、运行环境（Python 版本 + 全部依赖及版本）、运行命令、文件说明。

## 4. 当前状态、分数与天花板

**当前交付 v5**：`weight_tol` 树 + 3 seed TCN（42/43/44，各 150 epoch）状态字典平均；
逐目标融合 POR 树0.45/TCN0.55、PERM 0.05/0.95、SW 0.15/0.85；分类概率 = 树0.45 + TCN0.55；
阈值 POR 0.12 / PERM 0.07 / SW 0.04。

| 口径 | A（v4 同源基线 120ep×2） | v5（150ep×3） | Δ |
|---|---|---|---|
| 固定 v4 权重/阈值 screen | 68.576 | 68.710 | +0.134（三目标同向） |
| 可部署标量 `clf_tcn_w` screen | 68.662 | 68.753 | +0.091 |
| LOO（7 井调参→留出井，逐井等权） | 75.157 | 75.209 | +0.052（boot95 [−0.14,+0.24]，**跨 0，不构成证据**） |

v5 真实段 acc：**POR 0.1696 / PERM 0.0726 / SW 0.1427**；默认段 acc ≈ 0.988–0.994。

**距离 75 线的现实判断（重要）：**
- 默认段已达 98.8–99.4%，几乎无剩余空间 → **全部增益必须来自真实段**（当前仅 7%–17%）。
- 已测天花板：纯默认输出 69.2 → oracle 分段+树 69.963 → +TCN 70.319 → mean 70.570 → **oracle 逐段均值 73.414** → 真段均值 oracle ≈75.4 → 持久性 y[t−1] oracle ≈87（**需真实历史目标值，推理时不可达**）。
- 结论：75 分线位于"纯测井信息"天花板的边缘。**纯日志 + 分段这条路线在 73.4 附近就撞顶**，要稳定超过 75 需要引入日志之外的信息或结构性更强的归纳偏置。任何提案先对照这张天花板表判断是否有 6+ 分空间，再动手。
- 天花板成因：段 level 从测井聚合特征 R²≈0.62；段内 wiggle 同频测井 R² 0.52–0.65；容差与段内波动同量级。

**不要误读**：`eval_deployed.py` 报告的 70.940 是用含验证井的全量模型自评，属**泄漏**，不能当基线。
`models/` 当前工件是全量 80 井训练的 v5，不要用它去自评 screen 井。

## 5. 硬约束（违反会段错误 / 挂起 / 打包不合格）

1. **同进程禁止 torch 与 sklearn `fit` 共存**（macOS OpenMP 双运行时 → 无 Python 栈的 SIGSEGV / 挂起，顺序敏感）。
   sklearn 与 torch 均懒加载（`src/model.py`、`solution/tcn_model.py` 用模块 `__getattr__`）。
   **拟合树的脚本禁止 `import experiments/run_tcn.py` / `run_tcn2.py`**（它们顶层 import torch）；需要其函数就本地复制（参见 `experiments/eval_v5_deployable.py`）。
2. 全量训练只走 `train.py`（`stage=all` 子进程串行 tree→tcn1→[tcn2]→assemble，每进程单一运行时）。
   设备 `--device mps` 快（≈10 s/epoch），`cpu` 稳但约 4× 慢。崩溃换设备重试。
3. **numpy 前向与 torch 的 parity 三要点**：GELU 用 tanh 近似、逐点 LayerNorm、残差加在 gelu 之后。
   改网络后必须跑 `experiments/parity_test.py`（要求误差 < 1e-4；v5 实测 ≈7.7e-7）。
4. **TCN 选 epoch 必须用"验证集调优后总分"，禁止用 val@0.5**（会选中偏向默认输出的早期权重）。
5. `train.py` 的 `train_tcn()` 恒返回**末轮 EMA**（无 epoch 选择）。因此可部署成员必须"末轮≈best"
   （150ep 的 T15/T16 满足，差 0.003–0.009；200ep 的 T12 差 0.131 → **不可部署**）。
6. 打包与校验只用 `tools/package_submission.py`。
7. **后台进程会随工具 session 结束被杀**：训练用前台 `exec_command` + `write_stdin` 轮询（≥5 min/次），
   **禁用 nohup**。断点查询用 `grep -c '^epoch' <log>`（勿用 `tee | grep` 组合）。
8. 每次 assemble 后核对 `model.pkl`：`len(tcn_sds) == seed 数`，且**无 `tcn2_sd`**
   （历史事故：只要 `.staging/tcn2.joblib` 存在就会被拼进 pass2，现已由 `--with_pass2` 门控）。

## 6. 评估协议（不可漂移）

- **screen 切分**：`pd.unique(logId)` 前 72 井训练、后 8 井验证；阈值在验证集 0.05–0.95 网格调优。
- **报告必须含分解**：各目标 acc / acc_def / acc_real（真实段 acc 是核心信号）。
- **晋级规则**：screen 增益 ≥ **+0.05** 且三目标方向一致；或 5 折 CV ≥ **+0.3** 且 ≥4/5 折非负。
- 融合权重 / 阈值只在验证集内调；**跨工件比较必须先固定同一套 w/th**。
- **screen 单切分噪声约 ±0.3**；小于此的增益必须用多 seed / LOO / bootstrap 复核，否则诚实标注为"噪声内"。
- 每个实验追加一行到 `experiments/results.csv` 并更新 `EXPERIMENTS.md`。

## 7. 已否决方向（勿重试，均已实测）

200ep 单族（末轮非最优，不可部署）、纯 200+200 集成；逐井 pct-rank、段级池化决策、三目标共识决策；
clf 融合变体（prob/logit/max/min/median）；第 4 seed（收益在噪声内）；p_def 温度/isotonic（单调重参数化，
数学上无增益）；推理后平滑、段内 β 收缩、Viterbi/中值滤波分段；1/|y| 反向加权；三目标 all_log；
期望分数混合决策；段级 HistGB + 线性 wiggle 层级模型；段级 kNN level；逐井 NN 迁移；
自回归 pass2 通道（三路 ≤ 两路）；GRU / AR-GRU；CASE 列。
历史：物理公式约束、预测后平滑、级联模型、TV 去噪。
（完整推导见 `EXPERIMENTS.md` 与 `HANDOFF_V5.md` §8。）

## 8. 常用命令

```bash
P=/opt/anaconda3/bin/python        # Python 3.13.9, torch 2.14.0, scikit-learn 1.7.2, MPS 可用

$P experiments/run_cv.py --mode screen --tag X --context 1 --priors 1 --mono 1     # 树系评估
$P experiments/run_cv.py --mode cv --folds 5 --tag X_cv --context 1 --priors 1 --mono 1
$P experiments/eval_v5_deployable.py                                                  # v5 族综合评估（screen+LOO+bootstrap）
$P experiments/run_tcn.py --tag T17_seed42_180ep --epochs 180 --steps 60 --seed 42   # 新 TCN
$P experiments/parity_test.py                                                         # numpy/torch parity

# 端到端交付（约 1.5–2 h，MPS）
$P train.py --data_dir ./train --out_dir ./models --with_tcn 1 --epochs1 150 \
    --seed_list 42,43,44 --per_target_w 0.55,0.95,0.85 --thresholds 0.12,0.07,0.04 \
    --clf_tcn_w 0.55 --device mps
$P predict.py --data_dir ./test --output ./result.json
$P tools/package_submission.py --workdir .    # 生成两个 zip + 跑合规 checklist + 解包烟雾预测
```

无正式测试框架；校验手段是 `tools/package_submission.py` 的 schema checklist、`parity_test.py`、以及 `experiments/` 下的评估脚本。

## 9. 仓库布局与文件整理原则

**分三层，性质完全不同：**

1. **规范交付层**（zip 的内容，必须干净、自洽、可复现）：
   `README.md predict.py train.py config.yaml requirements.txt src/ models/ examples/`
2. **研究脚手架层**（实验与决策依据，是 git 里的历史，不要删）：
   `experiments/`（评估脚本 + `results.csv` + `EXPERIMENTS.md`）、`HANDOFF_V5.md`、`PROJECT_SUMMARY.md`、`OPTIMIZATION_PLAN.md`、`PROMPT.md`
3. **可再生的中间产物**（已在 `.gitignore`，勿提交、勿当交付物）：
   `train/ test/ result.json *.zip artifacts*/ experiments/runs/ __pycache__/`

**整理原则：**
- `tools/package_submission.py` 是交付物的唯一真源；手改 zip 视为无效。
- `solution/`（含 `seq_model.py`/`ar_model.py`）与 `train_seq.py`/`train_ar.py` 是 GRU/自回归路线的**已失败历史**，不参与当前交付，但**不要删**——对应 §7 的否决记录。
- `PROMPT.md` / `OPTIMIZATION_PLAN.md` 是**早期（v1）文档**，其中的分数（70.63 基线）与优先级已过时；权威现状以 `HANDOFF_V5.md` + `EXPERIMENTS.md` §v5 交付为准。
- 根目录 `artifacts*/`（4 个）与 `__pycache__/`、`.DS_Store` 是垃圾；清理前先 `git status` 确认未被跟踪。
- 目录总大小 351M 几乎全是 `experiments/runs/`（TCN joblib 可再生）与数据；**不要**把数据或 runs 加进 git。

## 10. 工作规范

- 每个里程碑 git commit，commit 信息带分数对比（如 `v5 交付: ... 68.710 (+0.134 vs A 68.576)`）。
- 报告分数时**必须标口径**（screen / cv / LOO / 可部署标量 / 固定权重），不同口径的数字不可直接相减。
- 诚实标注不确定性：CI 跨 0 或落在噪声带内的增益，明说"不构成证据"，不要用乐观口径。
- 交接文档要自包含：接手者只读 `HANDOFF_V5.md` + `EXPERIMENTS.md` §v5 交付 + `PROJECT_SUMMARY.md` 就能继续。
