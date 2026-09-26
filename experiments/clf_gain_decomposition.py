"""v8 候选增益分解：分类器效应 vs 阈值效应 vs 交互项（纯缓存，约 2 秒）。

为什么需要这个（2026-09-26 的决定性证据）：
  `tune_clf_metric.py` 报出「每目标取最优候选 + 各自最优阈值」相对 v7 值 +0.099 分，
  方向通过留一井 jackknife（[+0.062, +0.119] 不跨 0），字面上过了 §6 的晋级规则
  （+0.099 ≥ +0.05 且三目标同向）。但它落在 ±0.3 噪声带内，必须复核。

复核方法：把 (分类器, 阈值) 的两因子改动拆开成三块 ——
  分类器效应 = acc(cand, v7 阈值) − acc(bal, v7 阈值)     ← 只换分类器
  阈值效应   = acc(bal, 候选阈值) − acc(bal, v7 阈值)     ← 只换阈值
  交互项     = acc(cand, 候选阈值) − 前两项之和
若两个独立分量都为负、而交互项为正，则「增益」只在特定的 (更差分类器 × 更大阈值)
联合配置里存在，是小验证集上的联合过拟合，不构成可部署证据。

本脚本只读 clf_retrain_pred.joblib（由 tune_clf_metric.py --keep 产出），不拟合任何模型。

用法: python experiments/clf_gain_decomposition.py
"""
from __future__ import annotations
import os, sys, time
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.feature import default_mask, load_wells, make_targets
from src.utils import DEFAULT_VALUES, EPS, TARGETS, WEIGHTS

RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
BND = 657_489
V7 = {"blend_w": {"POR": 0.50, "PERM": 0.50, "SW": 0.75},
      "thresholds": {"POR": 0.130, "PERM": 0.270, "SW": 0.050},
      "clf_tcn_w": {"POR": 0.60, "PERM": 0.70, "SW": 0.30}}
# 每目标「最优候选 + 其最优阈值」，来自 experiments/runs/clf_retrain.json
PICK = {"POR": ("mw05", 0.360), "PERM": ("mw", 0.320), "SW": ("cross+reg", 0.070)}
T0 = time.time()


def acc(c, pred, true):
    p = np.asarray(pred, np.float64); t = np.asarray(true, np.float64)
    if c == "PERM":
        return np.maximum(0.0, 1.0 - np.abs(np.log10(np.maximum(p / np.maximum(t, EPS), EPS))))
    tol = 0.08 if c == "POR" else 0.05
    return np.maximum(0.0, 1.0 - np.abs(p - t) / (tol * (np.abs(t) + EPS)))


def main() -> None:
    ck = joblib.load(os.path.join(RUNS, "clf_retrain_inputs.joblib"))
    pc = joblib.load(os.path.join(RUNS, "clf_retrain_pred.joblib"))
    Z = np.load(os.path.join(RUNS, "tcn_all80_parts.npz"))
    ens = {c: np.float32(Z[f"reg_{c}"][BND:]) for c in TARGETS}
    pdf = {c: np.float32(Z[f"pdf_{c}"][BND:]) for c in TARGETS}

    df = pd.concat(load_wells("./train", has_labels=True), ignore_index=True)
    Y = make_targets(df); DM = default_mask(Y)
    Yv, Dv = Y.iloc[BND:], DM.iloc[BND:]

    tot_bal = tot_cand = 0.0
    print("=" * 104)
    print("增益分解：只换分类器 / 只换阈值 / 交互项（回归器、blend_w、clf_tcn_w 均固定为 v7）")
    print("=" * 104)
    print(f"{'目标':6s}{'bal@v7th':>10s}{'cand@v7th':>11s}{'bal@cTh':>10s}{'cand@cTh':>11s}"
          f"{'AUCbal':>8s}{'AUCcand':>9s}")

    for c in TARGETS:
        y = Yv[c].to_numpy(); m = ~np.isnan(y)
        rt = np.asarray(ck["rt_all"][c][BND:], np.float64)
        regb = np.power(10.0, (1 - V7["blend_w"][c]) * rt + V7["blend_w"][c] * ens[c])
        cw = V7["clf_tcn_w"][c]
        cand, th = PICK[c]
        th0 = V7["thresholds"][c]
        D = DEFAULT_VALUES[c]
        # 树 p_def 与 TCN pdf 的融合（cw = TCN 份额）
        b = (1 - cw) * pc[c]["bal"] + cw * pdf[c]
        q = (1 - cw) * pc[c][cand] + cw * pdf[c]
        A = lambda p: float(acc(c, p[m], y[m]).mean())
        fb = np.where(b >= th0, D, regb)
        fq = np.where(q >= th0, D, regb)
        fb2 = np.where(b >= th, D, regb)
        fq2 = np.where(q >= th, D, regb)
        clf_eff = A(fq) - A(fb)
        th_eff = A(fb2) - A(fb)
        inter = A(fq2) - A(fb2) - clf_eff
        ab, ac = A(fb), A(fq2)
        tot_bal += WEIGHTS[c] * ab * 100
        tot_cand += WEIGHTS[c] * ac * 100
        auc_b = float(roc_auc_score(Dv[c].to_numpy().astype(int)[m], pc[c]["bal"][m]))
        auc_q = float(roc_auc_score(Dv[c].to_numpy().astype(int)[m], pc[c][cand][m]))
        print(f"{c:6s}{ab:10.4f}{A(fq):11.4f}{A(fb2):10.4f}{ac:11.4f}{auc_b:8.4f}{auc_q:9.4f}")
        print(f"      分类器效应（只换分类器） {clf_eff:+.4f}   阈值效应（只换阈值 "
              f"{th0:.3f}→{th:.3f}） {th_eff:+.4f}   交互项 {inter:+.4f}"
              f"   AUCΔ {auc_q-auc_b:+.4f}")
        if clf_eff <= 0 and th_eff <= 0:
            print(f"      → 两个独立分量都为负，净收益全部来自交互项：联合过拟合，不可部署")
        elif clf_eff > 0:
            print(f"      → 分类器本身有正效应（AUC 同向），是唯一非过拟合的部分")

    print("-" * 104)
    print(f"  bal@v7 阈值（本 harness 复算）= {tot_bal:.3f}   "
          f"（部署 v7 = 73.749，差 {tot_bal-73.749:+.3f}，来自 POR 回归器 ~7e-3 log10 残差）")
    print(f"  cand@候选阈值                = {tot_cand:.3f}   Δ = {tot_cand-tot_bal:+.3f}")
    print(f"  screen 噪声 ≈ ±0.3；本次 |Δ| = {abs(tot_cand-tot_bal):.3f} "
          f"{'在噪声带内' if abs(tot_cand-tot_bal) < 0.3 else '超出噪声带'}")
    print(f"  耗时 {time.time()-T0:.1f}s")


if __name__ == "__main__":
    main()
