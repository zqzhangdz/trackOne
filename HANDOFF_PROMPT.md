# 交接 Prompt：赛道一「储层参数 POR/PERM/SW 智能预测」继续优化

你是接手续做者。仓库在 `/Users/yourmac/Downloads/THUNDER/赛道一`（git main 分支，16 commits）。
Python 用 `/opt/anaconda3/bin/python`（含 sklearn/torch/scipy；系统 python3 无依赖）。
先读三个文件建立上下文：`PROJECT_SUMMARY.md`（全量总结）、`EXPERIMENTS.md`（实验台账与事故记录）、
`experiments/results.csv`（全部实验结果）。本 prompt 已自包含开工所需的全部硬约束。

## 1. 任务与数据

- 输入：14 列测井曲线（`DEPTH,GR,PE,SP,CAL,AC,DEN,CNL,RXO,RT,DEVI,AZIM,BIT,CASE`），逐深度点 0.1 m；
  `train/` 80 井 730,268 点（含 POR/PERM/SW 标签），`test/` 10 井 95,948 点（无标签）。
  文件前 2 行是表头/单位，`read_csv(skiprows=2)`。CASE 不可用（仅 1 口训练井有值、测试井全缺），别碰。
- 哨兵值 `-99999 / -9999 / -2.534573822542048e37` 及 |x|>1e5 → NaN；CNL 截断 [0,100]。
- 目标结构：默认段（POR=0.1 / PERM=0.01 / SW=99.9，三默认同现 65.9%）与真实段二分；
  真实段沿深度高度平滑（lag-1 自相关 0.967），段间方差占 92–99%。
- 指标（ε=1e-5，逐点截断准确率均值）：
  `Acc_POR=mean(max(0,1-|ŷ-y|/(0.08(|y|+ε))))`；PERM 同式但在 log10 空间且分母为 `0.08(|log10 y|+ε)`；
  `Acc_SW` 容差 0.05；`Total=(0.3·Acc_POR+0.35·Acc_PERM+0.35·Acc_SW)×100`。实现见 `src/utils.py:acc_single`。
- 提交：`result.zip`（result.json：`{modelId,modelName,version,resultData:[{logId,predictions:[{depth,POR,PERM,SW}]}]}`，depth 升序）
  + `submission_code.zip`（目录严格为 README.md/predict.py/requirements.txt/config.yaml/train.py/
  src/{model,feature,utils}.py/models/{model.pkl,scaler.pkl}/examples/result_example.json）。
  复现命令 `python predict.py --data_dir ./data --output ./result.json`。打包只用 `tools/package_submission.py`（自带 checklist）。

## 2. 当前状态（v3，已交付）

- 架构：① 逐目标 HistGB 分类器（默认/真实，上下文特征）② 真实段 HistGB 回归（PERM log10 空间；POR 单调约束 DEN−/AC+；
  **样本权重 ∝ 指标容差宽度**：POR/SW 用 |y|、PERM 用 |log10 y|，5–95 分位截断）③ 双向 TCN（ch128、dilation 1–16×2、
  120 epoch、EMA、截断相对误差损失）④ log10 空间融合 **树 0.3 / TCN 0.7**；p_def ≥ 阈值（0.10–0.15/0.05/0.05）输出默认常数。
  推理纯 numpy（`src/model.py:numpy_forward_tcn`），提交包不依赖 torch。
- 分数：screen(72/8 切分) **68.545**；5 折 CV：树 M1 = 72.18±1.51（基线点式树 70.63±1.43）；v3 CV 估计 ≈72.7（非直接测量）。
  真实段 acc：POR 0.173 / PERM 0.065 / SW 0.148。
- 上界参照：纯默认 69.2；**真段均值 oracle ≈75.4**（=75 分线量级，纯测井信息天花板）；持久性 y[t-1] oracle ≈87（需真实历史目标，不可达）。
  天花板成因：段水平 level 从测井聚合特征 R²≈0.62；段内 wiggle 同频测井 R² 0.52–0.65；容差与段内波动同量级。

## 3. 硬工程约束（违反会段错误/挂起/打包不合格）

1. **同一进程禁止"torch + sklearn fit"共存**（macOS OpenMP 双运行时，顺序敏感、无 Python 栈的 SIGSEGV/挂起）。
   `src/model.py` 中 sklearn 与 torch 均懒加载（`get_tcn_classes()`/`_clf/_reg` 内部 import）；
   `solution/tcn_model.py` 用模块 `__getattr__` 懒加载。**拟合树的脚本禁止 import `experiments/run_tcn.py` 或 `run_tcn2.py`**
   （它们顶层 import torch）；需要 pointwise/decompose 就本地复制（见 ensemble_screen.py/eval_hybrid.py 的做法）。
2. 全量训练走 `train.py`（stage=all 自动子进程串行 tree→tcn1→[tcn2]→assemble，每进程单一运行时）。
   TCN 训练设备：`--device mps` 快（~15 s/epoch）但 MPS 驱动有过不可复现 SIGSEGV；`cpu` 稳但 ~4× 慢。崩溃时换设备重试。
3. numpy 前向与 torch 的 parity 三要点：GELU 用 tanh 近似、逐点 LayerNorm（不是 GroupNorm(1,C)）、残差加在 gelu 之后。
   改动网络后必须跑 parity 测试（误差 < 1e-4）。
4. TCN 选 epoch 必须用"验证集调优后总分"，**禁止用 val@0.5**（会选中偏向默认输出的早期权重）。
5. 打包/校验只用 `tools/package_submission.py`；zip 顶层不得多文件；predict 烟雾测试必须通过。
6. 每个里程碑 git commit；实验结果追加到 `experiments/results.csv` 并更新 `EXPERIMENTS.md`。

## 4. 评估协议

- screen 切分：`pd.unique(logId)` 前 72 井训练、后 8 井验证；阈值在验证集上按 0.05–0.95 网格调优；
  报告必须含分解：各目标 acc / acc_def / acc_real（真实段 acc 是核心信号）。
- 晋级规则：screen 增益 ≥ +0.05 且三目标方向一致，或 5 折 CV ≥ +0.3 且 ≥4/5 折非负；融合权重/阈值只在验证集内调。
- 噪声意识：screen 单切分噪声 ±0.3，微小增益需多 seed 或 CV 复核再采纳。

## 5. 禁止重试的已否决方向（均有台账记录）

推理后平滑、段内 β 收缩、Viterbi/中值滤波分段、1/|y| 反向加权、三目标 all_log、期望分数混合决策、
段级 HistGB+线性 wiggle 层级模型、段级 kNN level、逐井 NN 迁移、自回归 pass2 通道（三路≤两路）、GRU/AR-GRU、CASE 列。

## 6. 建议的下一步（按性价比，附估计）

1. **3-seed TCN 集成**（seed 42/43/44 各 120ep，回归 log10 平均）：+0.1~0.3，成本 ~1.5 h（MPS）。
2. **p_def 温度校准/isotonic 后重调阈值**：回收默认段 ~1.2% 误判的一部分：+0.1~0.2，成本 ~20 min。
3. **段对齐池化头**：TCN 增加 segment-pooling 分支（按预测段池化日志特征预测 level，与点级偏差相加）：+0~0.3，成本 ~2 h；
   注意先例：显式段聚合 HistGB 曾输给隐式卷积，须以 screen 分解表判定。
4. 逐目标融合权重（POR/PERM/SW 各自网格）：+0~0.1，成本 ~30 min。
5. 更大通道 ch192 / 200 epoch：+0~0.2，成本 ~1 h。
采纳任一后：全量重训（`train.py --with_tcn 1 --epochs1 <选定>`）→ predict → package_submission → commit → 更新台账。

## 7. 完成定义与汇报格式

- 完成定义：新工件入 `models/`；`result.zip`、`submission_code.zip` 重新生成且 checklist 全绿；
  `EXPERIMENTS.md`/`results.csv` 更新；git 提交信息含分数对比。
- 汇报格式（每次实验）：
  `| 配置 | screen total | POR/PERM/SW real | acc_def 均值 | 结论(采纳/否决+原因) |`
  以及最终交付时的版本对比表（基线 70.63 CV / v2 68.284 / v3 68.545 screen / 你的版本）。
- 若所有候选方向增益均 < +0.05：停止优化，出具"天花板确认报告"（引用第 2 节上界数据），保持 v3 交付不变。

## 8. 常用命令

```bash
P=/opt/anaconda3/bin/python
$P experiments/run_cv.py --mode screen --tag X --context 1 --priors 1 --mono 1        # 树系评估
$P experiments/run_cv.py --mode cv --folds 5 --tag X_cv --context 1 --priors 1 --mono 1
$P experiments/eval_hybrid.py --t3_tag T6_long --weight_tol 1                          # 融合权重评估
$P experiments/run_tcn.py --tag T8_seed43 --epochs 120 --steps 60 --seed 43            # 新 TCN seed
$P train.py --data_dir ./train --out_dir ./models --with_tcn 1 --epochs1 120 --device mps
$P predict.py --data_dir ./test --output ./result.json
$P tools/package_submission.py --workdir .
```
