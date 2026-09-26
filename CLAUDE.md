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

## 2. 评分标准（官方口径，2026-09-26 已用公式原文核验）

ε = 1e-5，逐深度点独立、等权：

```
Acc_POR  = mean( max(0, 1 − |ŷ − y| / (0.08 · (|y| + ε))) )
Acc_PERM = mean( max(0, 1 − |log10(max(ŷ / y, ε))|) )        ← 无容差系数、无分母
Acc_SW   = mean( max(0, 1 − |ŷ − y| / (0.05 · (|y| + ε))) )
Total    = (0.30·Acc_POR + 0.35·Acc_PERM + 0.35·Acc_SW) × 100
```

- POR / SW 是**相对误差**容差（8% / 5%，随 |y| 缩放）；
  **PERM 是比值容差，恒为 1.0 decade，与真值大小无关**。
- 实现：`src/utils.py` 的 `acc_single` / `total_score`（官方口径）。
  `acc_single_legacy` 保留 2026-09-26 之前的旧 PERM 口径，仅供追溯历史数字。
- **不要在脚本里就地改公式**；需要新变体时单独建脚本并注明。
- ⚠️ `train.py` 的 TCN 损失 `metric_loss` 与 `--weight_tol` 的样本加权都按旧口径写的，
  详见 §4.5 —— **重训前必读**。

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

## 4. 当前状态、分数与天花板（2026-09-26 官方公式核验后重写）

**⚠️ 本节全部数字为官方口径。** 2026-09-26 之前仓库内的 PERM 公式是错的，导致本项目所有
分数系统性偏低约 4.5 分，并据此得出过"75 分线不可达、别再训练"的**错误结论**。
旧口径数字一律标 `旧`，仅供追溯。详见 §2 与 §11。

**当前交付 v7**：成员与 v5/v6 完全相同（`weight_tol` 树 + 3 seed TCN 42/43/44 × 150 epoch），
**唯一改动是官方口径下重调 PERM 阈值 0.065 → 0.27**（零 GPU）。v7 已提交（`5ef3a76`），
`result.zip` / `submission_code.zip`（version 7.0）均已生成：

| 字段 | v6 | **v7（当前）** |
|---|---|---|
| `blend_w`（树 + 3×TCN 等分） | 0.54 / 0.96 / 0.80 | **0.50 / 0.50 / 0.75** |
| `thresholds` | 0.115 / 0.065 / 0.040 | **0.130 / 0.270 / 0.050** |
| `clf_tcn_w` | 逐目标 {0.60, 0.75, 0.30} | **{0.60, 0.70, 0.30}** |

备份：`models/model.pkl.v6bak`（v6 配置）、`model.pkl.v5bak`（v5 配置），两级可回滚。
`src/model.py:151` 早已支持 dict 或标量。

**v7 = 73.749，距 75 分线 +1.251。** 见 §4.4 与 §4.10。

### 4.1 v6 分数与分解（screen 8 井，逐点等权）

| | **官方口径** | 旧口径（作废） |
|---|---|---|
| **Total** | **73.310** | 68.778 |

**距 75 分线 = +1.690，不是旧结论说的 +6.2。** 这正是"别的评测结果比我高很多分"的来源。

| 目标 | 权重 | acc | 默认段占比 | acc_def | **acc_real** | 对总分贡献 |
|---|---|---|---|---|---|---|
| POR | 0.30 | 0.6609 | 0.6003 | 0.9881 | **0.1694** | 19.827 |
| PERM | 0.35 | 0.7608 | 0.6063 | 0.9941 | **0.4015** | 26.628 |
| SW | 0.35 | 0.7673 | 0.7340 | 0.9929 | **0.1450** | 26.858 |

PERM real 从旧的 0.0727 变成 **0.4015**（同一次预测，只换公式）。**PERM 不再是瓶颈。**

### 4.2 真正的瓶颈：分段分类器，不是回归器

**这是 2026-09-26 的决定性发现。** 把 v6 的回归器与融合权重原封不动，只把"学习到的分段"
换成**真值分段**，同一批成员就能从 73.310 提到 **76.308**（`experiments/runs/screen_tree_cache.joblib`
+ `tcn_screen_outputs.npz`，screen 8 井）：

| 目标 | v6 现部署 acc | 贡献 | 真值分段 + 同回归器 | 贡献 | **分段器可追** |
|---|---|---|---|---|---|
| POR | 0.6609 | 19.827 | 0.6882 | 20.645 | **+0.818** |
| PERM | 0.7608 | 26.627 | 0.8072 | 28.253 | **+1.625** |
| SW | 0.7673 | 26.857 | 0.7831 | 27.410 | **+0.553** |
| **Total** | **73.310** | | **76.308** | | **+2.997** |

**回归器基本没问题**：PERM 点级 HistGB（72 井训练 / 8 井验证）real acc **0.5088**（纯树）
～ 0.5131（最佳 blend），完全够用。损失几乎全在"把哪些点判成默认段"这一步。
v6 的 PERM 分类器在阈值 0.065 下判 77.81% 的点为默认段，而真值只有 60.63% —— 判默认过度。

**但 76.308 不是可达上界**：分类器 AUC 只有 POR 0.8958 / PERM 0.8915 / SW 0.9727，
**信息不足**，不是阈值没调好。而且树的 AUC 与 TCN 的几乎相同（0.8966 vs 0.8935），
所以现有两路融合**无法靠融合提高 AUC**。要提高，必须换/改分类器本身。

| 目标 | AUC(树) | AUC(TCN) | AUC(v6 融合) | v6@th 判默认占比 | 真值 |
|---|---|---|---|---|---|
| POR | 0.8966 | 0.8935 | 0.8958 | 0.7536 | 0.6003 |
| PERM | 0.8906 | 0.8909 | 0.8915 | 0.7781 | 0.6063 |
| SW | 0.9701 | 0.9722 | 0.9727 | 0.8626 | 0.7340 |

三个目标都是"判默认过度"（预测占比远高于真值），PERM 最严重（+17.2 pp）。
上表与上表全部数字由 **`experiments/diag_seg_bottleneck.py`**（纯缓存，1 秒）复现。

### 4.3 分解口径陷阱

`acc_def` 用 `default_mask`（`isclose(y, 默认值)`）算出来**恒为 1.0000**，
对应 `acc_real` = 0.1515 / 0.3924 / 0.1253（POR/PERM/SW）。
而 §4.1 表里的 0.9881 / 0.9941 / 0.9929 是用"部署模型判默认"的掩码算的，
对应 `acc_real` = 0.1694 / 0.4015 / 0.1450。

两种拆法的 `acc` 与 `Total` **完全一致（都是 73.310）**，只是 real/def 边界取法不同。
**引用 `acc_real` 时务必标明用的是哪种掩码。**

### 4.4 零训练已榨到的极限

官方口径下逐目标联合扫 `(blend_w, threshold, clf_tcn_w)`（w 步 0.05 / th 步 0.02 / cw 步 0.10）：

| 目标 | v6 配置 | v6 acc | 最优配置 | 最优 acc | Δ贡献 |
|---|---|---|---|---|---|
| POR | w 0.54, th 0.115, cw 0.60 | 0.6609 | w 0.50, th 0.13, cw 0.60 | 0.6607 | −0.006 |
| **PERM** | w 0.96, th 0.065, cw 0.75 | 0.7608 | **w 0.50, th 0.27, cw 0.70** | **0.7738** | **+0.456** |
| SW | w 0.80, th 0.040, cw 0.30 | 0.7673 | w 0.75, th 0.05, cw 0.30 | 0.7670 | −0.012 |
| **Total** | | **73.310** | | | **73.749（+0.439）** |

**PERM 阈值 0.065 → 0.27 是唯一实质零训练收益（+0.456）**：0.065 是按旧 PERM 口径调的，
旧口径下"把真实段误判成默认"代价极小，所以阈值压得很低；官方口径下 1.0 decade 容差很宽松，
最优阈值高得多。

改完后距 75 仍差 **+1.251**，全部要靠**更好的分类器**（或重训 TCN）拿到。

### 4.5 训练侧确实错了（用户判断正确）

两处编码了旧目标：

1. `train.py::metric_loss` 的 PERM 分支：`d / (0.08·|log10 y|)` 再 `clamp(max=2.0)`，
   等效阈值 0.02–0.16 decade，**比官方严 6～50 倍**。
   **已修正**为 `clamp(max=1.0)`（`train.py:57-64`）。改动后重训的 TCN 与 v6 工件不可比。
2. `train.py:269-281` 的 `--weight_tol` 与 `FINAL_THRESHOLDS`：阈值/加权都是按旧口径定的。
   `weight_tol` 实测对官方 acc **零影响**（4 种 POR 加权方案全 0.5933、PERM 0.7608×3、
   SW 0.7366×4），不动；但阈值必须由 §4.4 的最优值替代。
3. **分类器 `DEFAULT_CLF_PARAMS` 的 `class_weight="balanced"`** 从未按官方口径校核过 ——
   这是 §4.2 指出的瓶颈所在，也是最值得改的训练侧配置。

POR/SW 的 loss（`LN10·d/TOL` = `|ln(ŷ/y)|/TOL`）与官方 `|ŷ−y|/(TOL·|y|)` 在小误差下恒等、
大误差下略宽松 —— 方向正确，不动。

### 4.6 官方口径下重测仍无效（勿重试）

- **树回归器加权方案**（4 种）：官方 acc 零差异。
- **逐点选择器**：5 井拟合/3 井留出 74.134 < v6 80.972；72 井重训版 56.270 vs 其自建基线
  70.208（**−13.9**）。两版都在拟合井上不如留出井，在拟合噪声。
- **PERM 段级模型**（`tune_perm_level.py`，真值分段上界）：最佳 0.4492（纯测井 + 树预测 POR），
  **低于点级 HistGB 的 0.5088**。段级路线对 PERM 不适用，别再投。
- **PERM loss 换 log10-MAE**（`absolute_error` = 官方指标本身）：只值 +0.003 real，不是杠杆。
- 旧否决清单里"段级 HistGB + 线性 wiggle"在官方口径下**仍然无效**（段级 < 点级），维持否决。

### 4.7 下一步（按性价比排序）

1. **（零训练，✅ 已完成 = v7）** §4.4 最优配置已落地：PERM 阈值 0.065 → 0.27，**73.310 → 73.749**
   （+0.439）。提交 `5ef3a76`，工件与 zip 均已更新。见 §4.10。
2. **（✅ 已完成，结果是否定的）** 重训分类器，见 §4.11。**6 个候选方案全部无效**：
   去掉 `class_weight`、官方判错代价加权（两种指数）、跨目标 `p_def` 特征、本目标 TCN 回归+概率
   特征，AUC 移动 0.000～0.006；表面增益 +0.099 落在 ±0.3 噪声带内，且分解后 POR / PERM
   的**独立分量全为负**（收益只在「更差分类器 × 更大阈值」的联合配置里存在，是联合过拟合）。
   **未落地，维持 v7。别再重训分类器。**
3. **（约 100 min，3 seed，需先确认）** 按修正后的 `metric_loss` 重训 TCN。
   **不建议**：§4.11 已证明分类器 AUC 天花板 ~0.897 是**特征的信息上限**，不是训练方式问题；
   TCN 的 AUC（0.8909）与树（0.8966）几乎相同，重训 TCN 不会突破这个上限。
   且本机 16 GB 在训练时反复提示内存不足。**要跑必须先问。**
4. **POR / SW 回归器已到顶，别再投。**
5. **唯一真正未试的杠杆：新信息源（新特征）。** 见 §4.11 结尾的推断。

### 4.8 口径陷阱（务必遵守）

- **screen 下界 65.392，不是 69.2**（69.2 是全 80 井纯默认分）。screen 后 8 井默认段占比
  低 7–9 个百分点（POR 60.0 / PERM 60.6 / SW 73.4% vs 训练集 66.6/67.5/71.7）。
  `PROMPT.md` 里那组数是训练集统计，不是数据集特性。
- **screen 8 井含病态井** `c7611b0148bb4b`：POR 默认占比 0.001、**SW 全 NaN**，与其他 79 井不同源。
- **旧 LOO 75.157 / 75.209 已作废**：独立复算只有逐井等权 70.3 / 逐点等权 68.0 量级，出处不明。
  逐井等权比官方逐点等权系统性虚高 **+2.31 分**。
- **本项目从未产生过一个可辩护的 > 75 分数**（旧口径下从未 > 68.8）。
- **不要误读**：`eval_deployed.py` 报的 70.940 用含验证井的全量模型自评，属**泄漏**。
  `models/` 当前工件是全量 80 井训练的 v7，不要用它自评 screen 井。

### 4.9 原阻塞项已解除

原"评分公式未获外部核验"**已解除**：2026-09-26 已从赛事方提供的公式原文图片核验，
§2 的四条公式确认为官方口径。`src/utils.py` 的 `acc_single` / `total_score` 是权威实现，
`acc_single_legacy` 保留旧口径仅供历史数字追溯。

### 4.10 v7 落地与部署口径陷阱（2026-09-26）

v7 是**纯配置落地**（零 GPU）：`models/model.pkl` 只改 `blend_w` / `thresholds` / `clf_tcn_w`，
成员（`weight_tol` 树 + 3 seed TCN 150ep）与 v5/v6 完全相同，`predict.py` 未改。
由 `experiments/eval_v7_config.py --write` 写入（晋级证据也在该脚本内）。
备份链 `models/model.pkl.v6bak` → `model.pkl.v5bak`，两级可回滚（无 `.v7bak`）。

**`weight_tol` 是回归器的一部分，不是可关的开关**（`train.py:278-290`）。两阶段模型拟合完后，
每个目标的回归器会在**真实段**上重拟合一次，样本权重为 PERM `w = |log10(max(y, ε))|`、
POR / SW `w = |y|`，截断到 [p5, p95] 后按均值归一。任何想"复现部署回归器"的实验脚本都必须带上它。
`experiments/eval_v4.py:124` 带它，但 `tune_clf_metric.py` 初版没带 —— 与部署版回归器中位数差
0.07~0.08 个 log10，在它之上调出的阈值对部署工件无效。**踩过一次**：它把表面增益 +0.106 修正为
+0.099，并改变基线绝对水平（POR 0.6600→0.6611、PERM 0.7727→0.7739）。

**核验方法**：任何调阈值的 harness 都要打印"与 `screen_tree_cache` 的中位绝对差"，正常应为
float32 舍入量级（≈1e-8）。`tune_clf_metric.py` 已内置该自检（`自洽核对` 段）。修好后
PERM 1.7e-8、SW 3.1e-8；**POR 仍差中位 7.1e-3 log10**（≈1.6% 相对，仅 0.41% 的点 > 0.1 log10，
已排除 `transform_target`（`perm_log` 下 POR/SW 为恒等）与 `inverse_transform_target`），根因未追到，
但它使 harness 基线系统性高约 +0.013 分（73.762 vs 部署 73.749），不改变任何结论。

### 4.11 分类器重训实验（2026-09-26，否定结果，别再重试）

`experiments/tune_clf_metric.py`（6 候选 × 3 目标 = 15 次拟合，约 5.5 min）+
`experiments/clf_gain_decomposition.py`（纯缓存，1 s）。缓存
`runs/clf_retrain_inputs.joblib`（`ver=2`，回归器含 `weight_tol`）与 `runs/clf_retrain_pred.joblib`。

**AUC 天花板属于特征，不属于训练方式。** 6 个真正不同的候选
（`class_weight` balanced / None / 官方判错代价加权 ×2 指数 / 跨目标 `p_def` 特征 /
本目标 TCN 回归+概率特征）：

| 目标 | AUC 基线 `bal` | 候选 AUC 范围 | 最大 AUCΔ |
|---|---|---|---|
| POR | 0.8966 | 0.8915–0.8968 | +0.0002 |
| PERM | 0.8906 | 0.8868–0.8967 | +0.0061 |
| SW | 0.9701 | 0.9649–0.9736 | +0.0035 |

AUC 移动 0.000～0.006。TCN 的 AUC（0.8909）与树（0.8966）几乎相同 → §4.2 已证"两路融合提不了
AUC"，本节进一步证明**换训练方式也提不了**。

**表面增益 +0.099 是联合过拟合，不是分类器改进。** 每目标取最优候选 + 各自最优阈值
（POR `mw05`@0.360、PERM `mw`@0.320、SW `cross+reg`@0.070）得 73.762 → 73.862；
留一井 jackknife [+0.062, +0.119] 不跨 0、5/8 井为正，**字面上过了 §6 的晋级规则**。
但把 (分类器, 阈值) 的两因子改动拆开（`clf_gain_decomposition.py`，回归器与融合权重全固定）：

| 目标 | 只换分类器 | 只换阈值 | 交互项 | AUCΔ |
|---|---|---|---|---|
| POR | **−0.0007** | **−0.0064** | +0.0088 | −0.0030 |
| PERM | **−0.0005** | **−0.0012** | +0.0022 | +0.0056 |
| SW | +0.0003 | −0.0011 | +0.0017 | +0.0035 |

POR / PERM 的**两个独立分量都为负**：新分类器在 v7 自己的阈值上更差，用旧分类器把阈值移过去
也更差，收益只存在于特定的（更差分类器 × 更大阈值）联合配置里 —— 小验证集上的经典联合过拟合
指纹。只有 SW `cross+reg` 有正的独立分类器效应（+0.0003 acc = +0.011 分，AUC 同向 +0.0035），
是唯一非过拟合的部分，但太小，不值得为它改 `src/model.py::predict`。

**未落地，维持 v7。** 理由：
(a) |Δ| = 0.099 在 ±0.3 噪声带内，只占 75 分缺口 (+1.138) 的 8.6%；
(b) 三分之二的增益来自交互项，换到官方测试集不一定会复现；
(c) 落地要改 `train.py`（regret 权重需两遍拟合：先拟合回归器、算逐点判错代价、再重拟合分类器）
与 `src/model.py::predict`（SW `cross` 特征走不了 225 列的 X 路径，需先算完三个目标的基线 `p_def`
再拼 `[X, p_def_POR, p_def_SW]`），交付管道的复杂度与回归风险和收益严重不匹配。

**方法论注记（本轮两次踩坑，别再犯）：**
1. **漏 `weight_tol` → 阈值调在错误的回归器上**（§4.10）。初版报 +0.106，修正后 +0.099。
   任何调阈值脚本先跑 `自洽核对`，中位差不到 1e-6 就别信自己的阈值。
2. **行级 bootstrap 在本项目无效**：72,779 行被 8 口井高度聚类（单井内 POR lag-1 自相关 0.967），
   SE ∝ 1/√N 会低估方差约一个数量级。只有按井做留一 jackknife 是诚实的。**8 口井才是实验单元。**

**结论性推断（本轮最有价值的产出）：** 训练侧的所有杠杆已穷尽 ——
回归器（§4.6：段级 < 点级、加权零差异）、分段器（本节：AUC 天花板是特征的信息上限）、
TCN（AUC 与树相同，重训无望突破）、融合与阈值（§4.4 网格已穷尽，+0.439 已落地）。
**唯一未试的杠杆是新信息源（新特征），不是任何"训练得更好"的方案。** 按现有 13 条测井曲线的
信息量，75 分线估计仍不可达；要突破需引入题目允许的新数据来源。

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

v7 / v8 阶段追加（`experiments/tune_clf_metric.py` + `clf_gain_decomposition.py`，见 §4.11）：
- **重训分段分类器（6 种训练方案）**：AUC 移动仅 0.000～0.006（POR/PERM 0.8915–0.8968，
  SW 0.9649–0.9736）。**天花板是特征的信息上限，不是训练方式。**
- **官方判错代价加权**（`class_weight=None` + `w = |acc(判默认) − acc(判真实)|`，指数 1.0 与 0.5）：
  在 v7 自己的阈值上**比基线更差**（POR −0.0007、PERM −0.0005），只在同时大幅调高阈值时才"赢"
  （+0.0088 / +0.0022 的交互项）。典型联合过拟合，别当收益。
- **跨目标 `p_def` 特征 + 本目标 TCN 回归/概率作分类特征**：只有 SW 有正效应（AUC 0.9701→0.9736，
  +0.011 分），不足以承担改 `src/model.py::predict` 的部署风险。POR / PERM 上加这两组特征
  AUC 反而下降（0.8966→0.8915/0.8918，0.8906→0.8890/0.8868）—— 三目标分段信号并不共享。
- **行级 bootstrap 复核**：在本项目无效（72,779 行被 8 口井聚类，POR 单井 lag-1 自相关 0.967），
  方差被低估约一个数量级。只用留一井 jackknife。**8 口井才是实验单元。**

## 8. 常用命令

```bash
P=/opt/anaconda3/bin/python        # Python 3.13.9, torch 2.14.0, scikit-learn 1.7.2, MPS 可用

$P experiments/run_cv.py --mode screen --tag X --context 1 --priors 1 --mono 1     # 树系评估
$P experiments/run_cv.py --mode cv --folds 5 --tag X_cv --context 1 --priors 1 --mono 1
$P experiments/eval_v5_deployable.py                                                  # v5 族综合评估（screen+LOO+bootstrap）
$P experiments/run_tcn.py --tag T17_seed42_180ep --epochs 180 --steps 60 --seed 42   # 新 TCN
$P experiments/parity_test.py                                                         # numpy/torch parity

# 官方口径分数与瓶颈定位（均不训练，秒级～分钟级）
$P experiments/diag_official_and_clean.py    # 官方 vs 旧口径双算 + 标签哨兵普查
$P experiments/diag_seg_bottleneck.py        # 瓶颈定位: 73.310 vs 76.308 + 分类器 AUC（1 秒，核心脚本）
$P experiments/diag_official_ceiling.py      # 官方口径天花板 + 单目标追分敏感度
$P experiments/tune_official_perm.py         # 官方口径 (w, th, cw) 精调 → 73.749
$P experiments/tune_perm_deployable.py       # PERM 点级 HistGB（不需分段）
$P experiments/diag_loo_aggregation.py       # 逐井等权 vs 官方逐点等权
$P experiments/diag_level_ceiling.py         # 纯日志段水平上界（旧口径，已作废）
$P experiments/tune_v6_grid.py               # 逐目标 (w, th, cw) 精调 + LOO 复核（旧口径）
$P experiments/tune_clf_metric.py --keep     # 分段器重训 6 候选（约 5.5 min，纯 sklearn，见 §4.11）
$P experiments/clf_gain_decomposition.py     # 分类器/阈值/交互项三分解（1 秒，纯缓存）

# 端到端交付（约 1.5–2 h，MPS）：train.py 只能出标量 clf_tcn_w，逐目标 dict 须事后写入
$P train.py --data_dir ./train --out_dir ./models --with_tcn 1 --epochs1 150 \
    --seed_list 42,43,44 --per_target_w 0.50,0.50,0.75 --thresholds 0.13,0.27,0.05 \
    --clf_tcn_w 0.70 --device mps
$P experiments/eval_v7_config.py             # v7 融合配置晋级证据 + --write 写回（备份 .v6bak）
$P experiments/apply_v6_config.py --dry-run  # 预览 v6 融合配置（更早的落地脚本）
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
- **2026-09-26（本项目最严重的错误）**：§2 的 PERM 公式是错的。旧写法把 PERM 容差算成
  `0.08 · |log10 y|`（实际约 0.02–0.36 decade），而官方公式原文是
  `Acc_PERM = mean(max(0, 1 − |log10(max(ŷ/y, ε))|))` —— **恒为 1.0 decade 的比值容差，
  没有容差系数、没有分母**。旧口径比官方严 3～20 倍。后果：
  - v6 分数从 **68.778 → 73.310**（+4.53 分），距 75 的缺口从 +6.2 缩到 **+1.69**；
  - §4.2 "PERM 缺口是物理不可达"**作废** —— 那个结论的分母用的是旧容差；
  - §4.2 结尾"不要再跑新 TCN 训练"的决策依据失效；
  - 整个天花板分析（69.0–69.4）都建立在错口径上。
  已修 `src/utils.py`（`acc_single` 改官方口径、`acc_single_legacy` 留旧口径做追溯）与
  `train.py::metric_loss`；`solution/metric.py` 顶部加作废横幅。
  **教训：本项目此前从未见到过公式原文，全部常数靠人工转写 —— 见 §4.3，拿到原文是最高优先级。**
- **2026-09-26**：标签缺陷规模一开始被严重夸大。`src/feature.py::_is_sentinel` 只拦
  `abs > 1e5`，POR `-6000.336` / PERM `-6000.373` / SW `-5960.427` 等确实漏网，但**全项目只有
  7 个点**（POR 3 / PERM 2 / SW 2），不是初判的"6701 / 6711 / 16308"。那个大数字是把
  **未清洗的原始列**（含 `-99999` / `-9999` 哨兵）拿去做 `default_mask` 造成的计数错误。
  结论：标签污染对分数无影响，不需要修清洗逻辑。
