"""探测：逐井 log10 中位校正的**留一井**诚实收益（而非验证集自评上界）。

动机
----
diag_deployed_residual.py 显示 POR/PERM 的「逐井 log10 中位」跨井 std 远大于井内 std
(POR 0.8805 vs 0.3998)，看起来像井级系统性乘性偏差，diag_welllevel_corrector.py 正在
据此建模。但同一脚本的 D4（按**预测值**分箱乘法校准）对 POR/SW 给出 Δ=+0.0000，
只有 PERM 给出 +0.58（且是自评上界）。两者可能矛盾：跨井 std 大并不等于「按井分箱能
降条件方差」——若方差集中在某个预测区间内、井与井之间共享，按井校正同样无效。

本脚本在**8 口验证井**上做留一井（well 是本项目唯一诚实的实验单元），裁决三种校正：
  W1 按井分箱的条件方差 vs 全局方差（方差是否真的按井结构）
  W2 逐井 log10 中位校正：(a) 训练井全局中位（对照） (b) 井级统计量线性映射预测该井中位
  W3 按预测值分箱校正（对照 D4），同口径留一井化，看 +0.58 有多少是真的

口径
----
沿用 diag_deployed_residual.py 的部署同源混合预测（v7 口径），reg 在 log10 空间，
校正量为**乘性**（corrected = p / 10**off），与 D4 一致。只在**真值真实段**上算官方
口径 acc。不 import torch，不写盘，纯缓存读取。
"""
from __future__ import annotations

import gc
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.feature import load_wells, make_targets
from src.utils import TARGETS, WEIGHTS, EPS, DEFAULT_VALUES

RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
TOL = {"POR": 0.08, "PERM": 1.0, "SW": 0.05}
T108 = {"T15": "T15_seed42_150ep", "T16a": "T16_seed43_150ep", "T16b": "T16_seed44_150ep"}
V7 = {"blend_w": {"POR": 0.50, "PERM": 0.50, "SW": 0.75},
      "thresholds": {"POR": 0.130, "PERM": 0.270, "SW": 0.050},
      "clf_tcn_w": {"POR": 0.60, "PERM": 0.70, "SW": 0.30}}


def acc_vec(c, p, t):
    p = np.asarray(p, float)
    t = np.asarray(t, float)
    if c == "PERM":
        return np.maximum(0.0, 1.0 - np.abs(np.log10(np.maximum(p / np.maximum(t, EPS), EPS))))
    tol = TOL[c]
    return np.maximum(0.0, 1.0 - np.abs(p - t) / (tol * (np.abs(t) + EPS)))


def well_stats(va, ids, all_wells):
    """井级统计量：每条曲线的 均值/中位/std → 每口井一行。推理时无标签可算。

    用 all_wells（全 80 井）算，这样验证井的统计量也可得（推理时同样只有本井曲线）。
    """
    ids = list(ids)
    cols = [c for c in all_wells.columns if c not in ("logId", "DEPTH")]
    rows = []
    for w in ids:
        d = all_wells[all_wells["logId"] == w]
        r = []
        for c in cols:
            s = pd.to_numeric(d[c], errors="coerce").to_numpy(dtype=float)
            s = s[np.isfinite(s)]
            r += [s.mean() if len(s) else np.nan,
                  np.median(s) if len(s) else np.nan,
                  s.std() if len(s) > 1 else np.nan]
        rows.append(r)
    return np.array(rows, dtype=float)


def main() -> None:
    t0 = time.time()
    cache = joblib.load(os.path.join(RUNS, "screen_tree_cache.joblib"))
    tcn = {k: v for k, v in np.load(os.path.join(RUNS, "tcn_screen_outputs.npz")).items()}
    rt, pt = cache["reg"], cache["p_def"]

    wells = load_wells("./train", has_labels=True)
    ids = [w["logId"].iloc[0] for w in wells]
    all_wells = pd.concat(wells, ignore_index=True)
    va = pd.concat([w for w in wells if w["logId"].iloc[0] in set(ids[72:])], ignore_index=True)
    del wells
    gc.collect()
    va = va.sort_index().reset_index(drop=True)
    assert len(va) == 72779, len(va)
    Y = make_targets(va)
    well = va["logId"].to_numpy()
    uw = np.unique(well)

    S = well_stats(va, uw, all_wells)
    id2row = {w: i for i, w in enumerate(uw)}

    print("=" * 96)
    print(f"部署同源混合预测（v7 口径）  验证 {len(va):,} 点 / {len(uw)} 井  井级统计量 {S.shape[1]} 列")
    print("=" * 96)

    for c in TARGETS:
        w, cw = V7["blend_w"][c], V7["clf_tcn_w"][c]
        ens = np.mean([tcn[f"{T108[t]}__{c}__reg"] for t in T108], axis=0)
        reg = (1 - w) * rt[c] + w * ens
        reg = np.where(np.isfinite(reg), reg, -5.0)
        p = np.power(10.0, reg)
        y = Y[c].to_numpy().astype(float)
        dv = DEFAULT_VALUES[c]
        real = np.isfinite(p) & np.isfinite(y) & ~np.isclose(y, dv)
        lo = np.log10(np.clip(p, 1e-12, None))
        rr = lo - np.log10(np.clip(y, 1e-12, None))
        base = acc_vec(c, p, y)[real].mean()

        print(f"\n{'─' * 96}")
        print(f"目标 {c}   真实段 {real.sum():,} 点（占 {real.mean() * 100:.1f}%）   基线真实段 acc {base:.4f}")

        # ---- W1 按井分箱：条件方差是否真的按井结构 ----
        cond_var_well = float(np.mean([np.nanvar(rr[real & (well == w)])
                                       for w in uw if (real & (well == w)).sum() > 50]))
        fold_var = float(np.mean([np.nanvar(rr[m] - np.nanmedian(rr[m]))
                                  for w in uw
                                  if (m := real & (well == w)).sum() > 50]))
        med_w = np.array([np.nanmedian(rr[real & (well == w)]) for w in uw
                          if (real & (well == w)).sum() > 50])
        base_w = np.array([acc_vec(c, p, y)[real & (well == w)].mean() for w in uw
                           if (real & (well == w)).sum() > 50])
        vb = np.mean([np.nanvar(rr[real & (well == w)]) for w in uw
                      if (real & (well == w)).sum() > 50])
        print(f"  W1b 逐井 log10 中位: {' '.join(f'{m:+.3f}' for m in med_w)}"
              f"  (std {med_w.std():.3f})")
        print(f"      逐井真实段acc : {' '.join(f'{a:.3f}' for a in base_w)}"
              f"  (std {base_w.std():.3f})   井间方差占比 {vb and (1 - vb / max(np.nanvar(rr[real]), 1e-12)) * 100:.0f}%")
        print(f"  W1 全局 std {np.nanstd(rr[real]):.4f}"
              f"  → 按井分箱条件 std {np.sqrt(cond_var_well):.4f}"
              f"  → 减井内中位后 std {np.sqrt(fold_var):.4f}"
              f"   |·|中位 {np.nanmedian(np.abs(rr[real])):.3f}")

        # ---- W2 留一井：井级中位校正 ----
        for name, mode in (("W2a 全局中位(对照)", "global"), ("W2b 井级线性映射", "linear")):
            gains, offs = [], []
            for w in uw:
                m_te = real & (well == w)
                m_tr = real & (well != w)
                if m_te.sum() < 50 or m_tr.sum() < 500:
                    continue
                if mode == "global":
                    off = float(np.nanmedian(rr[m_tr]))
                else:
                    tr_ids = [i for i in uw
                              if (real & (well == i)).sum() > 50 and i != w]
                    Xtr, ytr = S[[id2row[i] for i in tr_ids]], \
                        np.array([np.nanmedian(rr[real & (well == i)]) for i in tr_ids])
                    ok = np.isfinite(Xtr).all(axis=1) & np.isfinite(ytr)
                    if ok.sum() < 6:
                        continue
                    xm = Xtr[ok] - np.nanmedian(Xtr[ok], axis=0)
                    Xm = np.nan_to_num(xm)
                    ym = ytr[ok] - ytr[ok].mean()
                    beta, *_ = np.linalg.lstsq(Xm, ym, rcond=None)
                    xw = np.nan_to_num(S[id2row[w]] - np.nanmedian(xm, axis=0))
                    off = float(ym.mean() + xw @ beta)
                corr = np.clip(p / (10.0 ** off), 1e-12, None)
                gains.append(acc_vec(c, corr, y)[m_te].mean() - acc_vec(c, p, y)[m_te].mean())
                offs.append(off)
            if gains:
                np_ = int(sum(1 for g in gains if g > 0))
                print(f"  {name}: 留一井 Δ真实段acc 中位 {np.median(gains):+.4f}"
                      f"  区间 [{np.min(gains):+.4f},{np.max(gains):+.4f}] 正号 {np_}/{len(gains)}"
                      f"  → 折算总分≈{np.median(gains) * 100 * WEIGHTS[c] * real.mean():+.3f} 分"
                      f"   (off 中位 {np.median(offs):+.3f}, 跨井 std {np.std(offs):.3f})")
            else:
                print(f"  {name}: 无有效折")

        # ---- W3 对照：按预测值分箱校正，留一井化 ----
        for nb in (11, 21):
            gains = []
            for w in uw:
                m_te = real & (well == w)
                m_tr = real & (well != w)
                if m_te.sum() < 50 or m_tr.sum() < 500:
                    continue
                bins = np.quantile(lo[m_tr], np.linspace(0, 1, nb + 1))
                corr = p[m_te].copy()
                for b in range(nb):
                    ib_te = (lo[m_te] >= bins[b]) & (lo[m_te] < bins[b + 1])
                    ib_tr = m_tr & (lo >= bins[b]) & (lo < bins[b + 1])
                    if ib_te.sum() > 5 and ib_tr.sum() > 30:
                        ratio = np.nanmedian(np.power(10.0, rr[ib_tr]))
                        corr[ib_te] = p[m_te][ib_te] / ratio
                te = m_te.nonzero()[0]
                gains.append(acc_vec(c, corr, y[te]).mean() - acc_vec(c, p[te], y[te]).mean())
            if gains:
                np_ = int(sum(1 for g in gains if g > 0))
                print(f"  W3 按预测值分箱 nb={nb:>2} 留一井: Δ真实段acc 中位 {np.median(gains):+.4f}"
                      f"  区间 [{np.min(gains):+.4f},{np.max(gains):+.4f}] 正号 {np_}/{len(gains)}"
                      f"  → 折算总分≈{np.median(gains) * 100 * WEIGHTS[c] * real.mean():+.3f} 分")

    print(f"\n用时 {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
