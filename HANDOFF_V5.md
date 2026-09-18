# 交接：赛道一 v5 判定（进行中，已中断）

最后更新：2026-09-18 20:03（用户主动中断以保存进度）
HEAD: `5dfe3e3`（LN10 常量修复）；工作区另有未提交脚本 + results.csv 追加（见下）。

## 0. 一句话状态

**v4 已交付且完好（桌面 result.zip / submission_code.zip / models/ 均未改动）。**
v5 方向的候选结论已基本查清：**200ep TCN 在 screen 上相对 120ep 同源基线有小幅正增益**
（+0.14~0.15，三目标同向），但 **95% 置信区间跨 0、且 train.py 缺少 epoch 选择机制**，
**尚未达到"可放心采纳"的程度**，因此保持 v4 交付不变。T13（第二个 200ep seed）训练到
40/200 epoch 时被用户中断。

## 1. 硬约束（务必遵守，违反会段错误/挂起/打包不合格）

1. **同进程禁止 torch + sklearn fit 共存**（macOS OpenMP 双运行时 SIGSEGV，无 Python 栈）。
   `src/model.py` 的 sklearn/torch 均为懒加载；`solution/tcn_model.py` 用模块 `__getattr__` 懒加载。
   **拟合树的脚本禁止 `import experiments/run_tcn.py` 或 `run_tcn2.py`**（顶层 import torch）。需要 pointwise/decompose 就本地复制。
2. 全量训练只走 `train.py`（`stage=all` 自动子进程串行 tree→tcn1→[tcn2]→assemble，每进程单一运行时）。
   TCN 设备 `--device mps` 快（~1.5 s/epoch @200ep 本机实测 0.10 ep/s ≈ 10 s/epoch），
   `cpu` 稳但 ~4× 慢。崩溃换设备重试。
3. numpy 前向与 torch parity 三要点：GELU 用 tanh 近似、逐点 LayerNorm、残差加在 gelu 之后。
   改网络必须跑 parity（<1e-4）。
4. **TCN 选 epoch 必须用"验证集调优后总分"，禁止用 val@0.5**（会选中偏向默认输出的早期权重）。
5. 打包/校验只用 `tools/package_submission.py`；zip 顶层不得多文件；predict 烟雾测试必须通过。
6. 每个里程碑 git commit；实验结果追加 `experiments/results.csv` 并更新 `EXPERIMENTS.md`。
7. **后台进程随工具 session 结束会被杀**：训练用 `exec_command` 前台 + `write_stdin` 轮询，
   **禁用 nohup**。轮询间隔建议 ≥5 min（训练 200ep 约 33-42 min）。

## 2. 评估协议（不可漂移）

- screen 切分：`pd.unique(logId)` 前 72 井训练、后 8 井验证。
- 阈值在验证集上 0.05–0.95 网格调优；报告必须含分解：各目标 acc / acc_def / acc_real。
- 晋级规则：screen 增益 ≥ **+0.05** 且三目标方向一致；或 5 折 CV ≥ +0.3 且 ≥4/5 折非负。
- **screen 单切分噪声 ±0.3**，微小增益需多 seed / LOO / bootstrap 复核。
- 融合权重/阈值只在验证集内调；不得使用含验证井的全量部署模型自评（详见 §4 陷阱）。

## 3. 本 session 核心发现：LN10 bug 导致"调优工件 ≠ 部署工件"

- commit `5dfe3e3` 之前，`experiments/run_tcn.py:31` 的 `LN10 = log10(e)*log(10) = 1.0`，
  而正确值应为 `ln(10) = 2.3026`。该常量用于把 log10 空间残差换算成线性相对误差，
  **只影响 POR/SW 的回归损失权重（被低估 2.3×）**，不影响 PERM。
- **`train.py` 一直用正确 LN10**（`solution/tcn_model.py` 内部），所以历史 screen 调优工件
  （T6_long / T8_seed43 / T9_seed44）与部署工件**不同源**；台账里的 v4「68.757」由 buggy 工件调出。
- 影响面：`argmax` 选 epoch 不受常数缩放影响，故**历史"选 epoch"结论仍有效**；
  但用 buggy 工件调出的**融合权重/阈值**与部署工件不匹配。
- `run_tcn.py` tuned 打印曾有 7.342 这类异常小值，原因是 T6/T8/T9 早于 commit `595eee8`（nanmean 修复），
  退化井（SW 全 NaN）污染均值使值缩小 100 倍；**已修复**。

### 3.1 关键数值（本 session 实测，用于判定）

**规范协议还原**：68.757 = buggy T6+T8 工件 + 固定部署权重/阈值
（w 0.60/0.95/0.75，th 0.10/0.05/0.05，clf_w 0.5）。用 `eval_sweep.py`（逐目标调优 w+th）
同组得 68.800。→ 二者差异来自"固定 vs 调优"，不是工件差异。

**固定 v4 权重/阈值、只换工件（正确 LN10）——判定"工件是否更好"的干净口径**：

| 工件族（正确 LN10） | total | 三目标 real 方向 |
|---|---|---|
| A: T10+T11（120ep，当前 train.py 同源代理） | **68.576**（基线） | — |
| C: T12+T10（200+120ep） | **68.729 (+0.153)** | 全正 (+0.0029/+0.0059/+0.0025) |
| B: T12+T11（200+120） | 68.719 (+0.143) | 全正 |
| T12 单（200ep） | 68.697 (+0.121) | SW≈0 |

- C 逐井 6/1 正负（剔 1 口 NaN），boot95 = **[−0.014, +0.397]**（跨 0）
- B boot95 = [+0.002, +0.346]
- LOO（7 井调参→留出井）：A 75.053 → B/D ≈75.19（gain +0.139，5/2 井正负）
- 逐目标调优后 in-sample：A 68.675 / B 68.791 / C 68.801 / D 68.766

**欠训假设已证实**：T12(200ep) tuned 最优选 **epoch 132**（非末轮）≈69.058，末轮 68.927；
自调优 POR real 0.2303（vs T10 0.2040，+0.026）。

**T13 中断时状态**：40/200 epoch 时 val@0.5=64.186 real=(0.224, 0.063, 0.148)（早期值不可用于选 epoch）。
**未生成 `experiments/runs/tcn_T13_seed43_200ep.joblib`**（工件只在训练结束落盘），
故 T13 目前**不可评估，需重训**。

## 4. 陷阱（务必避免）

- **`eval_deployed.py` 跑已部署 `models/model.pkl` 在 8 口 screen 验证井得 70.940 是泄漏**
  （部署模型在含这 8 井的全 80 井上训练）。**绝不可当作对比基线**。
- 用 buggy 工件（T6/T8/T9）调出的 w/th 与修复后工件混用会产生"虚假增益"。
  跨工件比较**必须固定同一套 w/th**，再单独在验证内调优。
- `train.py` 的 `train_tcn()` **无 epoch 选择、恒返回末轮 EMA**（见 `train.py:181`）。
  因此 `--epochs1 200` 全量训练会落到 epoch-200 权重，与 screen 选出的 132 有 ~0.13 差距。
  **采纳 200ep 前必须先解决该不匹配**（加 epoch 选择，或接受末轮并用 screen 实测末轮分数）。
- `models/.staging/` 曾因残留 `tcn2.joblib` 造成"三路拼接"交付事故（commit `48151d8` 已修）。
  改动 assemble 逻辑后必须验证 `model.pkl` 的 `sds` 长度 = seed 数。

## 5. 建议的下一步（按性价比）

1. **重训 T13（200ep, seed43）**，凑齐与 `train.py --seed_list 42,43` 完全对应的 200ep 双 seed 族：
   ```bash
   P=/opt/anaconda3/bin/python
   cd /Users/yourmac/Downloads/THUNDER/赛道一
   $P experiments/run_tcn.py --tag T13_seed43_200ep --epochs 200 --steps 60 --seed 43 2>&1 | tee experiments/runs/T13_seed43_200ep.log
   ```
   用时约 33–42 min（MPS）。**必须前台运行 + 轮询**。断点：`grep -c '^epoch' experiments/runs/T13_seed43_200ep.log` 应到 200，
   且结束后存在 `experiments/runs/tcn_T13_seed43_200ep.joblib`。
2. 评估 T12+T13（双 200ep）：
   ```bash
   $P experiments/eval_sweep.py --combos "T10_ln10,T11_seed43_ln10|T12_seed42_200ep,T13_seed43_200ep|T12_seed42_200ep,T11_seed43_ln10|T10_ln10,T11_seed43_ln10,T12_seed42_200ep,T13_seed43_200ep" --tags T10_ln10,T11_seed43_ln10,T12_seed42_200ep,T13_seed43_200ep --clf_ws 0.3,0.5,0.7
   ```
   （前向结果缓存至 `experiments/runs/tcn_screen_outputs.npz`，可秒级重复；仅等价池化决策已否决，勿再试）
3. **判定**：相对部署同源基线 A(68.576) 增益 ≥ +0.05、三目标同向、多数井非负 → 采纳 v5；
   否则出「天花板确认报告」，**保持 v4 交付不变**。
4. **若采纳 v5**：先解决 epoch 选择（见 §4 陷阱），然后
   ```bash
   $P train.py --data_dir ./train --out_dir ./models --with_tcn 1 --epochs1 200 \
       --seed_list 42,43 --per_target_w <screen 定值> --clf_tcn_w 0.5 --device mps
   $P predict.py --data_dir ./test --output ./result.json
   $P tools/package_submission.py --workdir .
   ```
   再 commit（信息含分数对比）+ 更新 `EXPERIMENTS.md` / `results.csv`。
5. commit 全部未提交脚本与 results.csv（本 session 已留档，见 §7）。

## 6. 天花板参照与版本对比

- 纯默认 69.2；oracle 分段+树 69.963；+TCN 70.319；mean 70.570；oracle 逐段均值 73.414；
  **真段均值 oracle 75.4 = 75 分线量级（纯测井信息天花板）**；持久性 y[t-1] oracle ≈87（需真实历史目标，不可达）。
- 天花板成因：段 level 从测井聚合特征 R²≈0.62；段内 wiggle 同频测井 R² 0.52–0.65；容差与段内波动同量级。
- 版本：基线 CV 70.63 → v2 68.284 → v3 68.545 → **v4 68.757（screen，buggy 口径）**。
  **注意**：v4 的 68.757 与修复后同源基线 68.576 **不可直接相减**（工件不同源，见 §3）。

## 7. 未提交文件清单（本 session 新增，已在磁盘）

`experiments/`：`eval_sweep.py`（组合×clf_w 扫描 + TCN 前向缓存）、`eval_ln10_consistency.py`、
`eval_deploy_consistency.py`、`loo_deploy_retune.py`、`eval_deployed.py`（泄漏用途，勿用于对比）、
`eval_t12_decide.py`、`eval_seedcombos.py`、`clf_combo.py`、`diag_consensus.py`、
`diag_pooled_decision.py`、`diag_segpool.py`、`pctrank_loo.py`、`pdef_pertwell.py`；
`experiments/results.csv` 新增行：
`LN10_buggy_v4_T6T843`(68.800)、`LN10_corr_T10T11`(68.658)、`LN10_corr_T10T11_v4w`(68.637)、
`LN10_4way`(68.783)；`T10_ln10`/`T11_seed43_ln10`/`T12_seed42_200ep` 由 run_tcn 自动入库。

## 8. 已否决方向（勿重试）

本 session 新增：逐井 pct-rank（LOO 1 口负）、段级池化决策、三目标共识决策、
clf 融合方式变体（prob/logit/max/min/median 全在噪声内）、更多 seed、p_def 温度/isotonic（数学无增益）。
历史：推理后平滑、段内 β 收缩、Viterbi/中值滤波分段、1/|y| 反向加权、三目标 all_log、
期望分数混合决策、段级 HistGB+线性 wiggle 层级模型、段级 kNN level、逐井 NN 迁移、
自回归 pass2 通道（三路 ≤ 两路）、GRU/AR-GRU、CASE 列。

## 9. 常用命令

```bash
P=/opt/anaconda3/bin/python
cd /Users/yourmac/Downloads/THUNDER/赛道一
$P experiments/run_cv.py --mode screen --tag X --context 1 --priors 1 --mono 1     # 树系评估
$P experiments/run_cv.py --mode cv --folds 5 --tag X_cv --context 1 --priors 1 --mono 1
$P experiments/eval_hybrid.py --t3_tag T6_long --weight_tol 1
$P experiments/run_tcn.py --tag T13_seed43_200ep --epochs 200 --steps 60 --seed 43
$P train.py --data_dir ./train --out_dir ./models --with_tcn 1 --epochs1 120 --device mps
$P predict.py --data_dir ./test --output ./result.json
$P tools/package_submission.py --workdir .
```
