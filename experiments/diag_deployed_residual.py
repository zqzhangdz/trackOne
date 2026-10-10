"""部署管线残差结构：一阶段混合预测的残差是否可用「预测值分箱校准」修正？

动机
----
已知（diag_realreg_headroom.py，真值分段）：残差 lag1 自相关 POR 0.966 / PERM 0.998 / SW 0.946，
但**平滑预测使 acc 变差**（最优收缩 α=0.00）。残差自相关高 ≠ 可平滑 —— 因为指标是逐点相对误差。

本脚本改用**部署同源**的混合预测（screen_tree_cache + 3 seed TCN，v7 口径），在验证集 8 井上
查残差的**条件结构**，回答三个部署相关的问题：
  D1 残差 vs 预测值：残差是否是预测值的确定性函数？→ 若是，按预测值分箱的校准即可部署
  D2 残差 vs 井：残差是否主要是井级常数？→ 若是，12 列 wellz 应已捕获（一阶段已用它）
  D3 残差分布形状：是重尾（少数极端点）还是整体偏移？→ 决定「修少数点」还是「整体校准」
  D4 直接落地：用**训练集同款**分箱校准系数作用于验证集是否可行 → 必须交叉验证，这里用
     验证集内部的分箱关系做**上界**估计（偏乐观），并明确标注

用法: python experiments/diag_deployed_residual.py
"""
from __future__ import annotations

import os
import sys
import gc
import time
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import numpy as np
import pandas as pd

from src.feature import load_wells, make_targets
from src.utils import TARGETS, WEIGHTS, EPS, DEFAULT_VALUES

RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
T108 = {"T15": "T15_seed42_150ep", "T16a": "T16_seed43_150ep", "T16b": "T16_seed44_150ep"}
V7 = {"blend_w": {"POR": 0.50, "PERM": 0.50, "SW": 0.75},
      "thresholds": {"POR": 0.130, "PERM": 0.270, "SW": 0.050},
      "clf_tcn_w": {"POR": 0.60, "PERM": 0.70, "SW": 0.30}}
TOL = {"POR": 0.08, "PERM": 1.0, "SW": 0.05}


def acc_vec(c, p, t):
    p = np.asarray(p, float)
    t = np.asarray(t, float)
    if c == "PERM":
        return np.maximum(0.0, 1.0 - np.abs(np.log10(np.maximum(p / np.maximum(t, EPS), EPS))))
    tol = TOL[c]
    return np.maximum(0.0, 1.0 - np.abs(p - t) / (tol * (np.abs(t) + EPS)))


def main():
    t0 = time.time()
    cache = joblib.load(os.path.join(RUNS, "screen_tree_cache.joblib"))
    tcn = {k: v for k, v in np.load(os.path.join(RUNS, "tcn_screen_outputs.npz")).items()}
    rt, pt = cache["reg"], cache["p_def"]

    wells = load_wells("./train", has_labels=True)
    ids = [w["logId"].iloc[0] for w in wells]
    va = pd.concat([w for w in wells if w["logId"].iloc[0] in set(ids[72:])], ignore_index=True)
    del wells
    gc.collect()
    va = va.sort_index().reset_index(drop=True)
    assert len(va) == 72779
    Y = make_targets(va)
    well = va["logId"].to_numpy()
    uw = np.unique(well)

    print("=" * 96)
    print(f"部署同源混合预测（v7 口径）  验证 {len(va):,} 点 / {len(uw)} 井")
    print("=" * 96)

    pred_real, pm, ytrue = {}, {}, {}
    for c in TARGETS:
        w, cw = V7["blend_w"][c], V7["clf_tcn_w"][c]
        ens = np.mean([tcn[f"{T108[t]}__{c}__reg"] for t in T108], axis=0)
        pcn = np.mean([tcn[f"{T108[t]}__{c}__pdf"] for t in T108], axis=0)
        pm[c] = (1 - cw) * pt[c] + cw * pcn
        reg = (1 - w) * rt[c] + w * ens
        reg = np.where(np.isfinite(reg), reg, -5.0)
        pred_real[c] = np.power(10.0, reg)
        ytrue[c] = Y[c].to_numpy().astype(float)

    # 自检
    tot = 0.0
    parts = []
    for c in TARGETS:
        th = V7["thresholds"][c]
        p = np.where(pm[c] >= th, DEFAULT_VALUES[c], pred_real[c])
        a = acc_vec(c, p, ytrue[c])
        a = np.where(np.isfinite(ytrue[c]), a, np.nan)
        tot += WEIGHTS[c] * np.nanmean(a)
        parts.append(f"{c} {np.nanmean(a):.4f}")
    print(f"\n[自检] Total = {tot * 100:.3f}   {'  '.join(parts)}   预期 73.749")

    for c in TARGETS:
        tol = TOL[c]
        p = pred_real[c]
        y = ytrue[c]
        ok = np.isfinite(p) & np.isfinite(y)
        # 真实段 = 真值非默认
        dv = DEFAULT_VALUES[c]
        real = ok & ~np.isclose(y, dv)
        print(f"\n{'─' * 96}")
        print(f"目标 {c}   真实段 {real.sum():,} 点（占 {real.mean() * 100:.1f}%）")

        rr = np.log10(np.clip(p, 1e-12, None)) - np.log10(np.clip(y, 1e-12, None))
        rp = p - y
        # D1 残差 vs 预测值
        lo = np.log10(np.clip(p, 1e-12, None))
        rb = np.quantile(lo[real], np.linspace(0, 1, 6))
        row = "  D1 按预测 log10 分箱： n / 残差log10 中位 / 残差绝对中位"
        for b in range(5):
            m5 = real & (lo >= rb[b]) & (lo < rb[b + 1])
            if m5.sum() > 100:
                row += (f"\n      [{rb[b]:+.2f},{rb[b + 1]:+.2f})  n={m5.sum():>7,}"
                        f"   中位 {np.nanmedian(rr[m5]):+.3f}"
                        f"   |·|中位 {np.nanmedian(np.abs(rr[m5])):.3f}")
        print(row)
        # 条件方差：残差在给定预测值下还剩多少方差
        cond_var = np.mean([np.nanvar(rr[real & (lo >= rb[b]) & (lo < rb[b + 1])])
                            for b in range(5) if (real & (lo >= rb[b]) & (lo < rb[b + 1])).sum() > 100])
        print(f"     条件方差 {cond_var:.5f} → 条件标准差 {np.sqrt(cond_var):.4f}  "
              f"（全局 {np.nanstd(rr[real]):.4f}）")
        # 相关
        cov = np.corrcoef(lo[real], rr[real])[0, 1]
        print(f"     corr(预测log10, 残差log10) = {cov:+.4f}")

        # D2 逐井残差
        med_w = np.array([np.nanmedian(rr[real & (well == w)]) for w in uw
                          if (real & (well == w)).sum() > 50])
        within = float(np.sqrt(np.nanmean(
            [np.nanvar(rr[real & (well == w)]) for w in uw if (real & (well == w)).sum() > 50])))
        print(f"  D2 逐井残差 log10 中位：跨井 std {med_w.std():.4f}  井内 std {within:.4f}  "
              f"极差 {med_w.max() - med_w.min():+.3f}")

        # D3 分布形状
        aq = np.quantile(np.abs(rp[real] / (np.abs(y[real]) + EPS)), [0.5, 0.9, 0.95, 0.99])
        frac_worst = (np.abs(rp[real]) / (np.abs(y[real]) + EPS) > 5 * tol).mean()
        inband = np.abs(rp[real]) <= tol * (np.abs(y[real]) + EPS)
        # 极端点贡献
        contrib = acc_vec(c, p[real], y[real])
        tot_real = contrib.sum()
        worst = np.argsort(contrib)
        top10 = contrib[worst[:int(0.1 * len(contrib))]].sum()
        print(f"  D3 相对误差 |·| 分位 p50 {aq[0] * 100:6.2f}%  p90 {aq[1] * 100:6.1f}%"
              f"  p95 {aq[2] * 100:7.1f}%  p99 {aq[3] * 100:8.1f}%")
        print(f"     带内 {inband.mean() * 100:5.1f}%   相对误差 >5×容差 {frac_worst * 100:5.1f}%"
              f"   最差 10% 的点贡献全部真实段 acc 的 {top10 / max(tot_real, 1e-9) * 100:4.1f}%")

        # D4 上界：若按预测值分箱做最优乘法校准（验证集内自评 → 偏乐观上界）
        # 校准量是**乘性**的：corrected = p / 该分箱内 p/y 的中位比值。线性残差不能与
        # 比值相除（旧版把 p-y 除以比值，量纲混错，输出恒为 0，已作废）。
        best_gain = 0.0
        base_real = acc_vec(c, p, y)[real].mean()
        for nb in (6, 11, 21):
            rb2 = np.quantile(lo[real], np.linspace(0, 1, nb + 1))
            corr = p.copy()
            for b in range(nb):
                mb = real & (lo >= rb2[b]) & (lo < rb2[b + 1])
                if mb.sum() > 50:
                    med_ratio = float(np.nanmedian(np.power(10.0, rr[mb])))
                    corr[mb] = p[mb] / med_ratio
            gain = acc_vec(c, corr, y)[real].mean() - base_real
            best_gain = max(best_gain, gain)
        print(f"  D4 按预测值分箱乘法校准（**验证集自评上界，偏乐观**）：最佳真实段 acc Δ = "
              f"{best_gain:+.4f} = {best_gain * 100 * WEIGHTS[c]:+.4f} 分"
              f" × 真实段占比 → 折算总分 ≈ {best_gain * 100 * WEIGHTS[c] * real.mean():+.4f} 分")

    print(f"\n用时 {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
