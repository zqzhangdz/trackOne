# 赛道一「储层参数（POR/PERM/SW）智能预测」项目总结

> 更新时间：2026-09-18 ｜ git：main 分支 15 次提交 ｜ 当前交付版本：v3

---

## 一、赛道任务说明

### 1.1 任务定义

由 14 条测井曲线（逐深度点，采样间隔 0.1 m，一口井一个 `.txt` 文件）回归三个储层参数：

| 目标 | 含义 | 默认常数值 | 默认段占比（训练集） |
|---|---|---|---|
| POR | 孔隙度 | 0.1 | 66.0% |
| PERM | 渗透率 | 0.01 | 66.9% |
| SW | 含水饱和度 | 99.9 | 70.1% |

数据呈"非储层默认段 / 储层真实段"二分结构：三目标同时退化为默认常数的深度点占 65.9%；
真实段三目标高度重叠（Jaccard > 0.85）且沿深度高度平滑（POR lag-1 自相关 0.967）。

- 训练集：80 口井 / 730,268 深度点；测试集：10 口井 / 95,948 深度点。
- 输入列：`DEPTH,GR,PE,SP,CAL,AC,DEN,CNL,RXO,RT,DEVI,AZIM,BIT,CASE`（CASE 缺失率 ~99%，仅 1 口训练井有值、测试井全缺，已剔除）。
- 哨兵值：`-99999 / -9999 / -2.534573822542048e37` 及 |x|>1e5 均视为缺失；CNL 物理截断 [0,100]。

### 1.2 评价指标（ε = 1e-5，逐点截断准确率取平均）

```
Acc_POR  = mean( max(0, 1 − |ŷ−y| / (0.08·(|y|+ε)) ) )
Acc_PERM = mean( max(0, 1 − |log10(max(ŷ,ε)) − log10(max(y,ε))| / (0.08·(|log10(max(y,ε))|+ε)) ) )
Acc_SW   = mean( max(0, 1 − |ŷ−y| / (0.05·(|y|+ε)) ) )
Total    = (0.3·Acc_POR + 0.35·Acc_PERM + 0.35·Acc_SW) × 100
```

要点：容忍带为**相对误差**（POR/PERM 8%、SW 5%），带内满分、带外线性衰减到 0；
PERM 在 log10 空间且分母含 |log10 y|（y≈1 的点容差趋近 0，几乎不可得分）；
默认段预测准确即满分，故默认/真实分段质量直接决定基线分数。赛事参考基线 75 分。

### 1.3 提交规范

- `result.zip`：内含 `result.json`，结构 `{modelId, modelName, version, resultData:[{logId, predictions:[{depth,POR,PERM,SW}]}]}`，depth 升序。
- `submission_code.zip`：目录必须为
  `README.md / predict.py / requirements.txt / config.yaml / train.py / src/{model,feature,utils}.py / models/{model.pkl,scaler.pkl} / examples/result_example.json`。
- 复现命令：`python predict.py --data_dir ./data --output ./result.json`；要求一次性无人工干预完成读取→预处理→加载→推理→生成。

---

## 二、任务时间线（全部任务）

| 阶段 | 任务 | 产出 / 结论 |
|---|---|---|
| P0 | 规则与历史成果梳理 | `PROMPT.md`（规则+成果 prompt 化）、`OPTIMIZATION_PLAN.md`（门控优化规划） |
| P1 | 基线诊断 | `diagnose_baseline.py`：分数分解（丢分集中在真实段）、oracle 上界（持久性 0.56–0.67）、决策规则对比（硬阈值优于期望分数混合） |
| P2 | M1 上下文化树基线 | lag/lead+窗口统计+物理先验特征、POR 单调约束；screen A–K 11 组实验；**5 折 CV 70.63→72.18** |
| P3 | M3 双向 TCN | `solution/tcn_model.py`+`experiments/run_tcn.py`；修复选 epoch 机制（val@0.5 会选中默认偏向的早期权重）→ T3(80ep) 真实段 0.197/0.085/0.156 |
| P4 | M4 自回归 pass2 | `run_tcn2.py`：pass1 因果平滑通道 + scheduled sampling；T5/T7；最终验证**无增益**，转为可选 |
| P5 | 混合融合与 v2 交付 | 树分类 + log10 空间多路回归融合；`predict.py` 混合推理；`tools/package_submission.py` 规范 checklist；v2 = 三路 0.4/0.3/0.3，screen 68.284 |
| P6 | 工程事故处理 | macOS sklearn/torch OpenMP 双运行时挂起/段错误 → 懒加载 + 子进程分阶段训练；numpy/torch 前向 parity 1e-6；GELU/GroupNorm/残差顺序对齐 |
| P7 | 信息天花板诊断 | `diag_segment/diag_wiggle/diag_transfer`：段间方差 92–99%、段均值 oracle 75.4、wiggle R² 0.52–0.65；层级模型/段 kNN/逐井迁移/β 收缩/CASE 全部否决 |
| P8 | M6–M7 最终优化 | metric-aware 容差宽度加权 +0.083；T6(160ep, 选 120)；两路 0.3/0.7 = **screen 68.545** |
| P9 | v3 交付 | 全量重训（tree+weight_tol → TCN 120ep）→ `result.zip`/`submission_code.zip` 重生成，全 checklist 通过，commit `cf6ef07` |

---

## 三、当前项目结构

```
赛道一/
├── README.md                  # 提交说明（算法/运行方式/依赖/指标）
├── predict.py                 # 主预测入口（混合推理，纯 numpy TCN 前向）
├── train.py                   # 复现训练入口（子进程分阶段: tree/tcn1/[tcn2]/assemble）
├── config.yaml                # 模型路径/名称版本/特征开关 (feature_context, feature_priors)
├── requirements.txt           # numpy/pandas/scikit-learn/joblib/PyYAML（推理不依赖 torch）
├── src/
│   ├── model.py               # 两阶段树模型 + 混合推理 + numpy TCN 前向 + torch TCN(懒加载)
│   ├── feature.py             # 读取/哨兵处理/点式+上下文+物理先验特征（单井安全）
│   └── utils.py               # 常量、官方指标、IO 工具
├── models/
│   ├── model.pkl              # v3 工件：树集成 + TCN(120ep) 权重 + blend_w[0.3,0.7] + 阈值
│   └── scaler.pkl             # 树特征列 + TCN 输入标准化统计
├── examples/result_example.json
├── result.json / result.zip   # v3 预测结果（10 井 95,948 点）
├── submission_code.zip        # v3 提交代码包（规范 12 路径）
├── tools/package_submission.py# 打包 + 规范 checklist + 解包烟雾预测 + schema 校验
├── experiments/               # 开发实验区（不进 zip）
│   ├── run_cv.py              # 统一 CV/screen 评估（树系）
│   ├── run_tcn.py / run_tcn2.py  # TCN pass1/pass2 训练（选 epoch 机制）
│   ├── ensemble_screen.py / eval_hybrid.py  # 融合权重评估
│   ├── diagnose_baseline.py / diag_segment.py / diag_wiggle.py / diag_transfer.py  # 诊断
│   ├── smooth_test.py / run_level.py         # 否决实验
│   ├── results.csv            # 全部实验结果台账
│   └── runs/                  # TCN 工件与训练日志（T2/T3/T5/T6/T7、pass1 缓存）
├── solution/                  # 历史模型代码（GRU/AR-GRU/旧两阶段；tcn_model 现为 src 别名）
├── diagnose_baseline.py       # 基线诊断（根目录保留）
├── PROMPT.md / OPTIMIZATION_PLAN.md / EXPERIMENTS.md / PROJECT_SUMMARY.md
├── train/ (80 井)  test/ (10 井)   # 赛方数据（git 忽略）
└── .git/                      # main 分支 15 commits
```

---

## 四、当前项目状态

### 4.1 分数台账（screen = 72/8 井切分；cv = 5 折 GroupKFold）

| 版本 | screen | 5 折 CV | 真实段 acc POR/PERM/SW | 说明 |
|---|---|---|---|---|
| 基线：点式两阶段树 | 67.330 | 70.63 ± 1.43 | 0.127/0.042/0.106 | 历史起点 |
| M1：上下文+先验+单调约束 | 68.020 | **72.18 ± 1.51** | 0.150/0.052/0.131 | 唯一完整 CV 验证 |
| v2：三路融合（树0.4/T3 0.3/T5 0.3） | 68.284 | ≈72.4（估计） | 0.161/0.059/0.138 | 已交付（被 v3 取代） |
| **v3：两路（树0.3/T6 0.7）+ metric-aware** | **68.545** | ≈72.7（估计） | 0.173/0.065/0.148 | **当前交付** |
| 参考上界：纯默认输出 | 69.20 | — | — | 默认段即满分 |
| 参考上界：真段均值 oracle | ≈75.4 | — | 0.178/0.311/0.233 | 纯测井信息天花板量级 |
| 参考上界：持久性 y[t-1] oracle | ≈87 | — | 0.668/0.598/0.560 | 需真实历史目标，不可达 |

### 4.2 交付物状态

- `submission_code.zip`（9.76 MB）：规范 11+1 文件齐备、顶层无多余项；解包后烟雾预测 10 井通过；推理零 torch 依赖。
- `result.zip`（0.58 MB）：10 井 95,948 点，schema/升序校验通过；三默认占比 0.764（逐井 0.65–0.85，井间异质合理）。
- git：15 次提交覆盖基线→M1→M3/M4→工程修复→v2→M6/M7→v3；数据与大型工件已 gitignore。
- 复现：`python train.py --data_dir ./train --out_dir ./models --with_tcn 1`（约 40 min，MPS）→ `python predict.py --data_dir ./test --output ./result.json`。

---

## 五、应用算法（v3 最终架构）

1. **分段判别（树）**：逐目标 `HistGradientBoostingClassifier`（上下文特征）判别默认/真实；
   特征 = 13 曲线 + logRT/logRXO/RT/RXO + 单井上下文（lag/lead ±1,2,3,5,10、窗口 5/11/21 均值/标准差、一阶差分、rel_depth）+ 物理先验（POR_D/POR_S/POR_avg/POR_ds）。
2. **真实段回归（树）**：`HistGradientBoostingRegressor`，PERM 在 log10 空间；POR 施加 DEN 单调减 / AC 单调增约束；
   **metric-aware 样本权重 ∝ 指标容差宽度**（POR/SW 为 |y|、PERM 为 |log10 y|，5–95 分位截断）——把容量集中在能得分的点。
3. **双向 TCN 回归**：非因果时序卷积（ch=128，dilation 1–16 ×2，感受野 ≈25 m），log10 空间三目标 + 默认 logit；
   损失 = 截断相对误差代理（带内线性/带外截断 + 0.05·L2）+ 默认 BCE；AdamW + cosine + EMA；120 epoch；
   推理为**纯 numpy 手写前向**（与 torch parity < 1e-6）。
4. **融合与决策**：树回归与 TCN 回归在 log10 空间按 **0.3/0.7** 加权；分段概率 ≥ 阈值（POR 0.10–0.15、PERM 0.05、SW 0.05，验证集调优）输出默认常数，否则输出融合值；输出 clip 到物理范围。
5. **工程结构**：train.py 以子进程分阶段执行（tree/tcn1/[tcn2]/assemble），隔离 sklearn 与 torch 的 OpenMP 运行时；sklearn/torch 在 src/model.py 内全部懒加载。

### 5.1 已验证否决的方向（含原因）

| 方向 | 结果 | 否决原因 |
|---|---|---|
| 1/\|y\| 样本加权 | 66.156 | 方向反了：过度关注小值点，POR 真实段崩溃 |
| Viterbi / 中值滤波结构化分段 | 67.668 / 67.880 | 分类器错误非孤立抖动；平滑以默认段 acc 换微小增益 |
| 推理后平滑 / 段内 β 收缩 | 单调变差 | 回归误差非"无偏噪声"型，平滑/收缩抹掉有效波动 |
| 三目标 all_log 变换 | 中性 | 不改变排序 |
| 期望分数混合决策 | 66.13 | 残差 σ 过大时退化为全默认 |
| 层级模型（段级 level + 点级 wiggle） | 67.99（50/50 融合） | 显式段聚合不如 TCN 隐式 level |
| 段级 kNN 非参 level | 0.026/0.014/0.124 | 聚合空间维数灾难 |
| 逐井 NN 迁移（同地层假设） | −0.04~−0.12 | 测试井无强同地层匹配；单井数据不足 |
| 自回归 pass2（因果平滑通道） | 三路 ≤ 两路 | 通道只携带 level 信息，wiggle 追踪未建立 |
| GRU / 自回归 GRU（历史） | 62–64 | 欠训练 + 阈值寻优与反馈回路不一致 |
| CASE 列 | 不可用 | 仅 1 口训练井有值、测试井全缺 |

---

## 六、缺陷与不足

### 6.1 效果层面（与 75 分线的差距）

1. **信息天花板**：段水平（level）从测井聚合特征的可解释方差仅 R²≈0.62，而指标容差极紧
   （POR 8%·|y|、PERM 8%·|log10 y|、SW 5%·|y|），段内波动 σ 与容差同量级。真段均值 oracle ≈75.4 说明
   75 分线≈"完美 level+wiggle 提取"，当前 level 估计误差是主要瓶颈；所有结构性路线（层级/kNN/迁移/AR）均撞同一面墙。
2. **PERM 最弱**（真实段 0.065）：其容差分母 |log10 y| 使 y≈1 的点几乎不可得分，且 log10 空间波动 σ≈0.29 ≫ 容差 0.08–0.16；测井信息对 PERM level 的解释力与 POR 相当但容差更紧。
3. **默认段仍丢 ~1.2%**（acc_def 0.988）：边界点误判直接送 0 分；结构化分段未能修复（错误为系统性边界误差）。
4. **验证协议偏弱**：融合权重、阈值、weight_tol 等决策多在单一 8 井 screen 切分上调优（噪声 ±0.3），
   仅 M1 做过完整 5 折 CV；v3 的 CV 72.7 为"tree CV + screen 增量"的估计值，非直接测量；存在对 screen 切分的轻度过拟合风险。
5. **TCN 选择协议不完美**：screen 工件用 8 验证井调优选 epoch；最终全量模型直接取末 epoch（cosine 收敛），未做 OOF 选择。

### 6.2 算法/建模层面

1. TCN 感受野 25 m < 平均段长 45–48 点（4.5–4.8 m？注：0.1 m 采样下 45 点 = 4.5 m，感受野覆盖足够），但**段对齐聚合**缺失：卷积共享权重、不按段边界池化，level 估计次优于理想段模型。
2. 损失为截断代理而非指标本身的可微形式；带外点梯度为 0，早期训练信号稀疏（靠 0.05·L2 维持）。
3. 无不确定性/校准层：p_def 校准差（调优阈值低至 0.05–0.10 反映），未做温度缩放或 isotonic 校准。
4. 未利用测试井的无标签信息（transductive/逐井分布对齐仅监控未启用）。
5. 单 seed TCN：未做多 seed 集成（估计 +0.1~0.3 的方差缩减空间）。

### 6.3 工程层面

1. **环境脆弱**：macOS 上 sklearn(libomp) 与 torch 同进程即挂起/段错误（顺序敏感、无 Python 栈），靠懒加载+子进程隔离规避；MPS 驱动曾出现不可复现 SIGSEGV；最终训练被迫在 MPS/CPU 间切换（CPU 太慢）。
2. 训练成本：全量 v3 约 40 min（MPS）；实验累计 GPU/CPU 时约 6+ 小时；无断点续训。
3. `solution/` 与 `experiments/` 存在历史冗余代码（GRU/AR-GRU/旧两阶段），仅 `src/`+`train.py`+`predict.py` 为提交链路；仓库根目录仍有散落诊断脚本。
4. 工件体积：model.pkl 14.9 MB（含 TCN 权重），评审环境加载内存峰值未做压测。
5. results.csv 为追加式台账，schema 随实验演进出现列补丁（restval 空值），可读性一般。

### 6.4 与赛事预期的差距归因（诚实结论）

- 75 分线对应"level+wiggle 完美提取"的信息天花板；当前 v3 提取到该天花板的约 55–60%（screen 68.545 vs 天花板 ≈75.4–76.7）。
- 差距主因不是工程或调参，而是**测井→储层参数的物理映射在本数据集上的信噪比**：段水平 R²≈0.62、段内 wiggle R²≈0.52–0.65，二者叠加后的可得分点比例即当前水平。
- 剩余低风险增量（估计合计 +0.3~0.8）：多 seed TCN 集成、更大通道/更长训练、逐目标融合权重、p_def 校准后重调阈值、段对齐池化头（segment-pooling head）再试。

---

## 七、可复现命令速查

```bash
# 评估（screen / 5 折）
python experiments/run_cv.py --mode screen --tag X --context 1 --priors 1 --mono 1
python experiments/run_cv.py --mode cv --folds 5 --tag X_cv --context 1 --priors 1 --mono 1
# 融合评估
python experiments/eval_hybrid.py --t3_tag T6_long --weight_tol 1
# 全量训练 + 预测 + 打包
python train.py --data_dir ./train --out_dir ./models --with_tcn 1 --epochs1 120 --device mps
python predict.py --data_dir ./test --output ./result.json
python tools/package_submission.py --workdir .
```
