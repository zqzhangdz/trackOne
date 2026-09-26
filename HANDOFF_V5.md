# 交接：赛道一 v5 已交付（2026-09-19，可安全接手）

> **⚠️ 已被 `HANDOFF_V6.md`（2026-09-26）取代 —— 请先读那一份。**
> 本文件保留作历史依据，但**至少四处结论已被推翻**，接手者切勿据此决策：
> 0. **【最严重，2026-09-26】** 本文件全部分数（68.576 / 68.710 / 68.753 等）用的是**错误的
>    PERM 评分公式**（容差 0.08·|log10 y|，官方实为恒 1.0 decade 比值容差）。
>    v6 官方口径是 **73.310**，距 75 仅 +1.69，不是这里的 +6.2。整个"已撞顶、别训练"的叙事
>    都建立在错口径上。见 `CLAUDE.md` §11。
> 1. §4 的 LOO 75.157 / 75.209 不可复现，且是**逐井等权**口径（比官方逐点等权虚高 +2.31）；
>    v6 官方口径 screen 只有 73.310。
> 2. §6 用真值段均值 oracle 推出"PERM 有 3.3 分可追"—— 纯日志只能做到该 oracle 的 16.7%，
>    且该 oracle 本身也是按旧 PERM 容差算的。
> 3. §5 第 1 条"需给 train.py 加 `--clf_tcn_w` 解析"已过时，`src/model.py:151` 早已支持 dict。
>
> 详见 `HANDOFF_V6.md` §2–§4 与 `CLAUDE.md` §2、§4、§11。

最后更新：2026-09-19 03:22 ｜ HEAD：`4e8f73d`（v5 交付 commit，现为 `ce9ae7b`）｜ 分支：main
本文件自包含，接手者只需读本文件 + `EXPERIMENTS.md` §v5 交付 + `PROJECT_SUMMARY.md`。

## 0. 一句话状态

**v5 已交付完成，全部工件就绪、checklist 全绿、已 commit。任务处于"可停止"状态。**
若要继续优化，见 §5 的候选方向（均已标注性价比与前置条件）。

## 1. v5 交付内容（当前最新）

- `models/model.pkl`：weight_tol 树 + **3 seed TCN（seed 42/43/44，各 150 epoch）状态字典平均**；
  逐目标融合权重 **POR 树0.45/TCN0.55、PERM 树0.05/TCN0.95、SW 树0.15/TCN0.85**；
  分类概率 = 树 0.45 + TCN 0.55 平均（`clf_tcn_w=0.55`）；阈值 **POR 0.12 / PERM 0.07 / SW 0.04**。
- `models/scaler.pkl`：TCN 标准化统计（80 井训练集 median/std）。
- `result.zip`：10 井 95,948 点，schema/深度升序校验通过，version 标签 5.0。
- `submission_code.zip`：11 个规范文件齐备、顶层无多余项、解包烟雾预测 10 井通过、零 torch 依赖。
- 复现命令（约 1.5–2 h，MPS）：
  ```bash
  P=/opt/anaconda3/bin/python
  $P train.py --data_dir ./train --out_dir ./models --with_tcn 1 --epochs1 150 \
      --seed_list 42,43,44 --per_target_w 0.55,0.95,0.85 \
      --thresholds 0.12,0.07,0.04 --clf_tcn_w 0.55 --device mps
  $P predict.py --data_dir ./test --output ./result.json
  $P tools/package_submission.py --workdir .
  ```

## 2. v5 分数与判定（诚实口径）

| 口径 | A（v4 同源基线：T10+T11, 120ep） | **v5（150x3: T15+T16a+T16b）** | Δ |
|---|---|---|---|
| 固定 v4 权重/阈值 screen | 68.576 | **68.710** | **+0.134**（三目标 real 全正） |
| 可部署标量 `clf_tcn_w` 调优 screen | 68.662（cw0.50） | **68.753**（cw0.55） | **+0.091** |
| LOO（7 井调参→留出井，逐井等权） | 75.157 | **75.209** | **+0.052**（boot95 [−0.140,+0.239]，4/3 井） |

- v5 real 段 acc：POR 0.1696 / PERM 0.0726 / SW 0.1427；acc_def ≈ 0.988–0.994。
- **判定**：固定权重口径满足晋级规则（≥+0.05 且三目标同向）→ **采纳 v5**。
  LOO 增益 +0.052 为正但 95% CI 跨 0，诚实标注为"处于 screen 单切分噪声带内"，非强证据。
- **关键设计**：`train.py` 的 `train_tcn()` 恒返回**末轮 EMA**（无 epoch 选择）。
  因此可部署成员必须"末轮≈best"。150ep 的 T15/T16a/T16b 满足（best 与末轮差 0.003–0.009）；
  200ep 的 T12 best@132 与末轮差 0.131 → 单靠它**不可部署**，故未采纳。

## 3. 硬约束（不可违反，违反会段错误/挂起/打包不合格）

1. **同进程禁止 torch + sklearn fit 共存**（macOS OpenMP 双运行时，顺序敏感、无 Python 栈的 SIGSEGV/挂起）。
   `src/model.py` 的 sklearn/torch 均懒加载；`solution/tcn_model.py` 用模块 `__getattr__` 懒加载。
   **拟合树的脚本禁止 `import experiments/run_tcn.py` / `run_tcn2.py`**（它们顶层 import torch）；
   需要 pointwise/decompose 就本地复制（见 `experiments/eval_v5_deployable.py`）。
2. 全量训练只走 `train.py`（`stage=all` 自动子进程串行 tree→tcn1→[tcn2]→assemble，每进程单一运行时）。
   TCN 设备 `--device mps` 快（≈10 s/epoch），`cpu` 稳但 ~4× 慢。崩溃换设备重试。
3. numpy 前向与 torch 的 parity 三要点：GELU 用 tanh 近似、逐点 LayerNorm、残差加在 gelu 之后。
   改网络后必须跑 `experiments/parity_test.py`（误差 < 1e-4）。v5 复测 max_err ≈ 7.7e-7。
4. **TCN 选 epoch 必须用"验证集调优后总分"，禁止用 val@0.5**（会选中偏向默认输出的早期权重）。
5. 打包/校验只用 `tools/package_submission.py`；zip 顶层不得多文件；predict 烟雾测试必须通过。
6. 每个里程碑 git commit；实验结果追加 `experiments/results.csv` 并更新 `EXPERIMENTS.md`。
7. **后台进程随工具 session 结束会被杀**：训练用 `exec_command` 前台 + `write_stdin` 轮询（≥5 min/次），
   **禁用 nohup**。断点查询用 `grep -c '^epoch' <log>`（勿用 `tee | grep` 组合，会让 write_stdin 无输出）。

## 4. 评估协议（不可漂移）

- screen 切分：`pd.unique(logId)` 前 72 井训练、后 8 井验证；阈值在验证集 0.05–0.95 网格调优。
- 报告必须含分解：各目标 acc / acc_def / acc_real（**真实段 acc 是核心信号**）。
- 晋级规则：screen 增益 ≥ **+0.05** 且三目标方向一致；或 5 折 CV ≥ +0.3 且 ≥4/5 折非负。
  融合权重/阈值只在验证集内调；跨工件比较必须先固定同一套 w/th。
- **screen 单切分噪声 ±0.3**；微小增益需多 seed / LOO / bootstrap 复核再采纳。
- **陷阱**：`experiments/eval_deployed.py` 用含验证井的全量模型自评（70.940）是**泄漏**，不可作基线。
- **陷阱**：`train.py` assemble 只要 `.staging/tcn2.joblib` 存在就会拼 pass2（历史交付事故，已由
  `--with_pass2` 门控修复）。每次 assemble 后核对 `model.pkl` 的 `len(tcn_sds) == seed 数`、无 `tcn2_sd`。
- 可信基线参考：`models/` 当前为 v5（全量 80 井训练），**不要**用它自评 screen 井。

## 5. 后续可继续的方向（按性价比；均已评估或标注前置条件）

1. **（低成本）逐目标 `clf_tcn_w`**：train.py 目前只接受标量 `--clf_tcn_w`。
   screen 显示逐目标 cw（POR0.6/PERM0.8/SW0.3）可将 PERM real 由 0.0726 提到 0.0722→略变，
   增益在噪声内；若要试，只需给 train.py 加 `--clf_tcn_w "0.6,0.8,0.3"` 解析（约 20 min，无重训）。
2. **（中成本）160–180ep 长日程 + 末轮部署**：150ep 已验证"末轮≈best"；更长日程可能再挤出 +0.05~0.1，
   但需重新确认末轮不劣化（必须打印逐 epoch tuned 曲线）。成本：3 seed ≈ 1.5–2 h（MPS）。
3. **（高成本、低确定性）段对齐池化头**：TCN 增加 segment-pooling 分支（按预测段池化日志特征预测 level，
   与点级偏差相加）。先例：显式段聚合 HistGB 曾输给隐式卷积，须以 screen 分解表判定。成本 ≈2 h。
4. **（高成本）ch192 / 200ep**：+0~0.2，成本 ≈1 h/seed。需先解决 epoch 选择（给 train.py 加 OOF 选择）
   或确认末轮≈best。
5. 采纳任一方向后的固定收尾流程：
   ```bash
   $P train.py --data_dir ./train --out_dir ./models --with_tcn 1 --epochs1 <N> \
       --seed_list <seeds> --per_target_w <w> --clf_tcn_w <cw> --thresholds <th> --device mps
   $P predict.py --data_dir ./test --output ./result.json
   $P tools/package_submission.py --workdir .
   # 更新 experiments/results.csv + EXPERIMENTS.md + PROJECT_SUMMARY.md，git commit（信息含分数对比）
   ```

## 6. 天花板参照与版本对比

- 纯默认输出 69.2；oracle 分段+树 69.963；+TCN 70.319；mean 70.570；oracle 逐段均值 73.414；
  **真段均值 oracle ≈75.4 = 75 分线量级（纯测井信息天花板）**；持久性 y[t-1] oracle ≈87（需真实历史目标，不可达）。
- 天花板成因：段 level 从测井聚合特征 R²≈0.62；段内 wiggle 同频测井 R² 0.52–0.65；容差与段内波动同量级。
- 版本谱系：基线 CV 70.63 → v2 68.284 → v3 68.545 → v4 68.757（screen，**buggy LN10 口径**）
  → **v5 68.753（可部署标量口径）/ 68.710（固定权重口径）**。
  **注意**：v4 的 68.757 与修复后同源基线 68.576 不可直接相减（工件不同源，见工程事故记录）。

## 7. 本 session 新增脚本（已 commit 于 `66f3ecb` 与 `4e8f73d`）

`experiments/eval_v5_deployable.py`（固定权重 + 调优 + 严格 LOO/bootstrap 综合评估；
已修复 `default_mask` KeyError 与 LOO 调参泄漏）、`experiments/eval_v5_fixed.py`（固定 v4 权重的族对照）、
`experiments/loo_v5_families.py`、`experiments/eval_t12_decide.py`（扩到 7 族）。
TCN 训练产物在 `experiments/runs/tcn_T1{3,4,5,6}*.joblib`（gitignore，可再生）。

## 8. 已否决方向（勿重试）

本 session：200ep 单族（末轮非最优，不可部署）、纯 200+200 集成（劣于 200+120）。
历史：逐井 pct-rank、段级池化决策、三目标共识决策、clf 融合变体（prob/logit/max/min/median）、
更多 seed（第 4 seed 收益在噪声内）、p_def 温度/isotonic（单调重参数化，数学无增益）、
推理后平滑、段内 β 收缩、Viterbi/中值滤波分段、1/|y| 反向加权、三目标 all_log、
期望分数混合决策、段级 HistGB+线性 wiggle 层级模型、段级 kNN level、逐井 NN 迁移、
自回归 pass2 通道（三路 ≤ 两路）、GRU/AR-GRU、CASE 列。

## 9. 常用命令

```bash
P=/opt/anaconda3/bin/python
cd /Users/yourmac/Downloads/THUNDER/赛道一
$P experiments/run_cv.py --mode screen --tag X --context 1 --priors 1 --mono 1     # 树系评估
$P experiments/run_cv.py --mode cv --folds 5 --tag X_cv --context 1 --priors 1 --mono 1
$P experiments/eval_v5_deployable.py                                               # v5 族综合评估（screen+bootstrap）
$P experiments/run_tcn.py --tag T17_seed42_180ep --epochs 180 --steps 60 --seed 42 # 新 TCN
$P experiments/parity_test.py                                                      # numpy/torch parity
$P train.py --data_dir ./train --out_dir ./models --with_tcn 1 --epochs1 150 \
    --seed_list 42,43,44 --per_target_w 0.55,0.95,0.85 --thresholds 0.12,0.07,0.04 \
    --clf_tcn_w 0.55 --device mps
$P predict.py --data_dir ./test --output ./result.json
$P tools/package_submission.py --workdir .
```
