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
  "version": "6.0",
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

**当前交付 v6**：与 v5 成员完全相同（`weight_tol` 树 + 3 seed TCN 42/43/44 × 150 epoch），
仅改融合配置（零 GPU 落地）：

| 字段 | v5 | v6 |
|---|---|---|
| `blend_w`（树 + 3×TCN 等分） | 0.55 / 0.95 / 0.85 | **0.54 / 0.96 / 0.80** |
| `thresholds` | 0.12 / 0.07 / 0.04 | **0.115 / 0.065 / 0.040** |
| `clf_tcn_w` | 标量 0.55 | **逐目标 {0.60, 0.75, 0.30}** |

v6 真实段 acc：**POR 0.1694 / PERM 0.0727 / SW 0.1450**；默认段 acc 0.9881 / 0.9941 / 0.9929。
备份：`models/model.pkl.v5bak`（v5 配置，可回滚）。逐目标 `clf_tcn_w` 无需改代码 ——
`src/model.py:151` 早已支持 dict 或标量。

### 4.1 分数口径修正（务必先读）

- **screen 下界是 65.392，不是 69.2。** `69.2` 是全训练集 80 井的纯默认分；screen 后 8 井
  默认段占比低 7–9 个百分点（POR 60.0% / PERM 60.6% / SW 73.4% vs 训练集 66.6/67.5/71.7），
  整条 screen 分数线相对更低的地板。`PROMPT.md` 里的 66.5/67.5/71.7% 是训练集统计，不是数据集特性。
- **screen 8 井含病态井** `c7611b0148bb4b`：POR 默认占比 0.001、**SW 全 NaN**，与其他 79 井不同源。
- **旧记录 LOO 75.157 / 75.209 已作废**：用同一批缓存独立复算得逐井等权 70.3 / 68.0 量级，无法复现，
  出处不明。且该口径是**逐井等权**，比官方逐点等权系统性虚高 **+2.31 分**。
- **本项目从未产生过一个可辩护的 > 75 分数。**

### 4.2 天花板重算：PERM 缺口是物理不可达

| 层次 | POR real | PERM real | SW real | total |
|---|---|---|---|---|
| 纯默认（下界） | — | — | — | 65.392 |
| **v6 当前** | 0.1694 | 0.0727 | 0.1450 | **68.778** |
| 纯日志段水平（HistGB 48 维段聚合，真值分段） | 0.1075 | **0.0520** | 0.1521 | 68.340 |
| 真值段均值 oracle | 0.1666 | 0.3110 | 0.2374 | 73.414 |
| 真值 oracle（不可达） | 1.0 | 1.0 | 1.0 | 100.0 |

`HANDOFF_V5.md` §6 曾用真值段均值 oracle（PERM 0.3110）推算"PERM 有 3.3 分可追"——**该推算错误**：
PERM 段水平 log10-RMSE = **0.4154 decade**，而官方相对容差是 8%（误差阈 = 0.08·|log10 y| ≈ 0.02–0.36 decade），
纯日志只能做到 0.0520
（oracle 的 16.7%；POR/SW 为 64.5% / 64.4%）。当前模型 PERM 0.0727 **已高于**纯日志段水平，
wiggle 捕获在净贡献；POR 同理（0.1694 > 0.1075）；**只有 SW 是 wiggle 净为负**（0.1450 < 0.1521，
SW 中位段长 15 m 短于 TCN 感受野 ~25 m，存在跨段边界混叠）。

**现实可达 ≈ 69.0–69.4；v6 的 68.778 约占 98%。** 距 75 分差 +6.2，实测上界只剩 +0.2～0.6，
差距约 10 倍。若测试集分布接近训练集，地板升至 69.204、现实上界约 72.7，仍差 2.3 分。
历史上唯一测到 > 75 的路径是持久性 y[t−1] oracle ≈87，需要真实目标历史值，推理时不可得。
**任何提案先对照这张表判断是否有 6+ 分空间，再动手；结论是不可能。**

**因此不要再跑新 TCN 训练（约 100 min/次）。** 成员不变的前提下，融合配置已在噪声带内榨干（§7）；
集成也已饱和。任何新训练的预期收益上限是 §4.2 里的 +0.2～0.6 分，对 75 分线的 +6.2 缺口无实质帮助，
而每次训练的成本是约 100 min。§8 的 `run_tcn.py` / `train.py` 命令保留是**为了可复现 v6 工件**，
不是为了继续优化。

### 4.3 待确认的阻塞项

**评分公式未获外部核验。** `参数说明.docx` 的四条公式是**图片**，图片托管 URL 已过期；
仓库内的容差 0.08/0.08/0.05、权重 0.30/0.35/0.35、ε=1e-5 全部来自人工转写。
若常数有误，本项目所有分数都会平移。**优先向赛事方索取公式原文** —— 这是唯一可能改变结论的外部信息。

**不要误读**：`eval_deployed.py` 报告的 70.940 是用含验证井的全量模型自评，属**泄漏**，不能当基线。
`models/` 当前工件是全量 80 井训练的 v6，不要用它去自评 screen 井。

## 5. 硬约束（违反会段错误 / 挂起 / 打包不合格）

1. **同进程禁止 torch 与 sklearn `fit` 共存**（macOS OpenMP 双运行时 → 无 Python 栈的 SIGSEGV / 挂起，顺序敏感）。
   sklearn 与 torch 均懒加载（`src/model.py`、`solution/tcn_model.py` 用模块 `__getattr__`）。
   **拟合树的脚本禁止 `import experiments/run_tcn.py` / `run_tcn2.py`**（它们顶层 import torch）；需要其函数就本地复制（参见 `experiments/eval_v5_deployable.py`）。
2. 全量训练只走 `train.py`（`stage=all` 子进程串行 tree→tcn1→[tcn2]→assemble，每进程单一运行时）。
   设备 `--device mps` 快（≈10 s/epoch），`cpu` 稳但约 4× 慢。崩溃换设备重试。
3. **numpy 前向与 torch 的 parity 三要点**：GELU 用 tanh 近似、逐点 LayerNorm、残差加在 gelu 之后。
   改网络后必须跑 `experiments/parity_test.py`（要求误差 < 1e-4；v6 实测 5.3e-7 / 7.7e-7 / 6.2e-7）。
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
- **报告分数必须标聚合方式**。逐井等权（`np.nanmean(逐井 total)`）比官方逐点等权系统性虚高 **+2.31 分**，两者不可互换引用（`experiments/diag_loo_aggregation.py`）。
- **screen 8 井不等于官方测试集分布**（默认段占比低 7–9 个百分点，含 1 口病态井）。任何"能否过 75"的结论都不能从 screen 分数直接外推，见 §4.1。
- 每个实验追加一行到 `experiments/results.csv` 并更新 `EXPERIMENTS.md`。

**注意 v6 是本规则的例外，且必须诚实标注**：v6 的 screen 增益 +0.025 低于 +0.05 晋级线，
按 §6 严格读**不构成晋级证据**。采纳理由是"实测不劣、配置更贴合逐目标最优、零 GPU 成本、可一键回滚"，
不是"确认有增益"。若有人要求严格证据，正确做法是回滚到 `models/model.pkl.v5bak`。

## 7. 已否决方向（勿重试，均已实测）

200ep 单族（末轮非最优，不可部署）、纯 200+200 集成；逐井 pct-rank、段级池化决策、三目标共识决策；
clf 融合变体（prob/logit/max/min/median）；第 4 seed（收益在噪声内）；p_def 温度/isotonic（单调重参数化，
数学上无增益）；推理后平滑、段内 β 收缩、Viterbi/中值滤波分段；1/|y| 反向加权；三目标 all_log；
期望分数混合决策；段级 HistGB + 线性 wiggle 层级模型；段级 kNN level；逐井 NN 迁移；
自回归 pass2 通道（三路 ≤ 两路）；GRU / AR-GRU；CASE 列。
历史：物理公式约束、预测后平滑、级联模型、TV 去噪。
（完整推导见 `EXPERIMENTS.md` 与 `HANDOFF_V5.md` §8。）

v6 阶段追加（`experiments/tune_v6_grid.py` + `runs/tune_v6_grid.json`）：
- **逐目标 (blend_w, threshold, clf_tcn_w) 精调再往下挖**：网格已穷尽（w 步长 0.02、th 步长 0.005、
  cw 步长 0.05），各候选族增益仅 +0.026～+0.040，全在噪声带内。**调参已榨干，别再调。**
- **更多 TCN 成员**：`150x3+T14` 68.781、`all6` 68.764，均不优于或仅平于 `150x3` 68.778。集成已饱和。
- **"用真值段均值 oracle 反推可追分数"**：`HANDOFF_V5.md` §6 据此算出"PERM 有 3.3 分可追"，
  已被 §4.2 实测证伪（纯日志段水平只能做到 oracle 的 16.7%）。别再用真值 oracle 当预算依据。

## 8. 常用命令

```bash
P=/opt/anaconda3/bin/python        # Python 3.13.9, torch 2.14.0, scikit-learn 1.7.2, MPS 可用

$P experiments/run_cv.py --mode screen --tag X --context 1 --priors 1 --mono 1     # 树系评估
$P experiments/run_cv.py --mode cv --folds 5 --tag X_cv --context 1 --priors 1 --mono 1
$P experiments/eval_v5_deployable.py                                                  # v5 族综合评估（screen+LOO+bootstrap）
$P experiments/run_tcn.py --tag T17_seed42_180ep --epochs 180 --steps 60 --seed 42   # 新 TCN
$P experiments/parity_test.py                                                         # numpy/torch parity

# v6 分数口径核对与天花板（新增，均不训练，秒级～分钟级）
$P experiments/diag_loo_aggregation.py       # 逐井等权 vs 官方逐点等权
$P experiments/diag_level_ceiling.py         # 纯日志段水平上界
$P experiments/tune_v6_grid.py               # 逐目标 (w, th, cw) 精调 + LOO 复核

# 端到端交付（约 1.5–2 h，MPS）：train.py 只能出标量 clf_tcn_w，逐目标 dict 须事后写入
$P train.py --data_dir ./train --out_dir ./models --with_tcn 1 --epochs1 150 \
    --seed_list 42,43,44 --per_target_w 0.55,0.95,0.85 --thresholds 0.12,0.07,0.04 \
    --clf_tcn_w 0.55 --device mps
$P experiments/apply_v6_config.py --dry-run   # 预览 v6 融合配置
$P experiments/apply_v6_config.py            # 写回 model.pkl（备份 .v5bak，失败自动回滚）
$P predict.py --data_dir ./test --output ./result.json
$P experiments/parity_test.py                 # numpy/torch parity（要求 < 1e-4）
$P tools/package_submission.py --workdir .    # 生成两个 zip + 跑合规 checklist + 解包烟雾预测
```

无正式测试框架；校验手段是 `tools/package_submission.py` 的 schema checklist、`parity_test.py`、以及 `experiments/` 下的评估脚本。

## 9. 仓库布局与文件整理原则

**分三层，性质完全不同：**

1. **规范交付层**（zip 的内容，必须干净、自洽、可复现）：
   `README.md predict.py train.py config.yaml requirements.txt src/ models/ examples/`
2. **研究脚手架层**（实验与决策依据，是 git 里的历史，不要删）：
   `experiments/`（评估脚本 + `results.csv` + `EXPERIMENTS.md`）、`HANDOFF_V6.md`（**当前权威**）、
   `HANDOFF_V5.md`（已标记作废结论，仅供追溯）、`PROJECT_SUMMARY.md`、`OPTIMIZATION_PLAN.md`、`PROMPT.md`
3. **可再生的中间产物**（已在 `.gitignore`，勿提交、勿当交付物）：
   `train/ test/ result.json *.zip artifacts*/ experiments/runs/ __pycache__/`

**整理原则：**
- `tools/package_submission.py` 是交付物的唯一真源；手改 zip 视为无效。
- `solution/`（含 `seq_model.py`/`ar_model.py`）与 `train_seq.py`/`train_ar.py` 是 GRU/自回归路线的**已失败历史**，不参与当前交付，但**不要删**——对应 §7 的否决记录。
- `PROMPT.md` / `OPTIMIZATION_PLAN.md` 是**早期（v1）文档**，其中的分数（70.63 基线）与优先级已过时；
  `HANDOFF_V5.md` 的分数口径（LOO 75.157/75.209）与天花板推算（PERM 3.3 分可追）已被推翻（§4.1、§4.2）。
  **权威现状以 `HANDOFF_V6.md` + `EXPERIMENTS.md` §v6 交付 + `CLAUDE.md` §4 为准。**
- 根目录 `artifacts*/`（5 个）、`__pycache__/`、`.DS_Store` 是可再生垃圾，**已于 2026-09-26 清理**；
  新增前先看是否已被 `.gitignore` 覆盖，确认未被 `git ls-files` 跟踪再删。
- 目录总大小 360M 左右，几乎全是 `experiments/runs/`（TCN joblib 可再生）与数据；**不要**把数据或 runs 加进 git。

## 10. 工作规范

- 每个里程碑 git commit，commit 信息带分数对比（如 `v6 交付: ... screen 68.778 (+0.025 vs v5 68.753)`）。
- **commit 末行必须带** `Co-Authored-By: Claude Code <noreply@anthropic.com>`；PR 描述末行带
  `🤖 Generated with [Claude Code](https://claude.com/claude-code)`。（本条曾被我漏掉，见 §11。）
- 报告分数时**必须标口径**（screen / cv / LOO / 可部署标量 / 固定权重），不同口径的数字不可直接相减。
- 诚实标注不确定性：CI 跨 0 或落在噪声带内的增益，明说"不构成证据"，不要用乐观口径。
- 交接文档要自包含：接手者只读 `HANDOFF_V6.md` + `EXPERIMENTS.md` §v6 交付 + `PROJECT_SUMMARY.md` 就能继续。

## 11. 本文件自身的已知错误记录

- **2026-09-26**：本文件初版（`3769eaf`）的 §4 沿用了 `HANDOFF_V5.md` 的分数口径，
  声称"纯日志+分段路线在 73.414 附近撞顶"，把**真值 oracle** 当成了可达上界。
  经 `diag_level_ceiling.py` 实测，纯日志段水平只能做到该 oracle 的 16.7%（PERM 0.0520 vs 0.3110），
  现实上界约 69.0–69.4。§4 已按实测重写。**教训：真值 oracle 只能当"不可达上界"，不能当预算依据。**
- **2026-09-26**：§5.8 的交叉引用已并入 §5 第 8 条，`apply_v6_config.py` 的报错提示指向 §5.8 时以本节为准。
