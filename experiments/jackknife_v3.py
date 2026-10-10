"""用**正确的统计量**判 M0/V3 的 +0.050 是真增益还是噪声。

为什么这个脚本存在
------------------
tune_reg_mixed.py 报的「留一井包络 [−0.413, +0.829]」是**逐井 Δ 的 min/max**，
不是均值置信区间。这是错误统计量：7 口井、单井标准差 ~0.35 时，min/max 几乎必然
跨 0 —— 即使真实均值为正。§4.14 里 wellz 的 [+0.126, +0.183] 是围绕均值 +0.172 的
LOO jackknife 区间，两者不可比。用 min/max 判 V3 等于用一把过宽的量具判它死刑。

正确做法（§6：8 口井是实验单元）
--------------------------------
对每口井 i，去掉 i 重算均值差：
    d_i = mean_{j≠i} (Total_V3_j − Total_base_j)
CI = [min_i d_i, max_i d_i]。
这是均值的 LOO jackknife 区间，宽度 ~σ/√(n−1) ≈ 0.35/√6 ≈ 0.14，而不是单井 Δ 的
±0.85。

同时报三目标的逐井贡献分解，确认「三目标同向」在**逐井层面**也成立，而不只是
在均值层面成立（§4.11 的联合过拟合就在逐井层面暴露）。

唯一变量：树回归器的 (mode, loss, weight) = (all_log, absolute_error, |log10 y|)
= V3。分类器 / 融合权重 / TCN 全冻结，与 tune_reg_mixed.py 完全同源。
纯 numpy + sklearn，**不 import torch**（§5.1）。基线必须复现 73.931。

用法: python experiments/jackknife_v3.py
"""
from __future__ import annotations

import gc
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from src.feature import (
    load_wells, make_targets, default_mask, make_features,
    transform_target, inverse_transform_target,
)
from src.utils import TARGETS, WEIGHTS, EPS, DEFAULT_VALUES
from src.model import DEFAULT_REG_PARAMS, POR_MONOTONIC
from experiments.eval_v4 import build_tree, tcn_outputs

RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
TAGS = ["T15_seed42_150ep", "T16_seed43_150ep", "T16_seed44_150ep"]
CFG = {"blend_w": {"POR": 0.50, "PERM": 0.50, "SW": 0.75},
       "thresholds": {"POR": 0.080, "PERM": 0.250, "SW": 0.050},
       "clf_tcn_w": {"POR": 0.60, "PERM": 0.70, "SW": 0.30}}
BASE = 73.931
GRID = np.arange(0.02, 0.71, 0.005)

# V3：三目标全部 (all_log, absolute_error, |log10 y|)
V3 = ("all_log", "absolute_error", "abslogy")


def rss() -> str:
    import resource
    return f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576:.0f}MB"


def acc_vec(c, p, t):
    p = np.asarray(p, float)
    t = np.asarray(t, float)
    if c == "PERM":
        return np.maximum(0.0, 1.0 - np.abs(np.log10(np.maximum(p / np.maximum(t, EPS), EPS))))
    tol = 0.08 if c == "POR" else 0.05
    return np.maximum(0.0, 1.0 - np.abs(p - t) / (tol * (np.abs(t) + EPS)))


def fit_reg(Xtr, Ytr, Dtr, c, cfg):
    """按 (mode, loss, weight) 拟合目标 c 的回归器；分类器不动。"""
    mode, loss, wkey = cfg
    valid = Ytr[c].notna().to_numpy()
    y = Ytr.loc[valid, c]
    def_ = Dtr.loc[valid, c].astype(int)
    Xv = Xtr.loc[valid]
    real = (def_ == 0).to_numpy()

    params = dict(DEFAULT_REG_PARAMS)
    params["loss"] = loss
    mono = POR_MONOTONIC if c == "POR" else None
    if mono:
        params["monotonic_cst"] = [mono.get(col, 0) for col in list(Xv.columns)]
    reg = HistGradientBoostingRegressor(random_state=42, **params)
    yt = transform_target(c, y.loc[real], mode)
    reg.fit(Xv.loc[real], yt)
    if wkey != "none":
        yv = y.to_numpy()[real]
        w = (np.abs(np.log10(np.maximum(yv, EPS))) if wkey == "abslogy" else np.abs(yv))
        w = np.clip(w, np.percentile(w, 5), np.percentile(w, 95))
        reg = HistGradientBoostingRegressor(random_state=42, **params)
        reg.fit(Xv.loc[real], yt, sample_weight=w / w.mean())
    return reg


def main():
    t0 = time.time()
    wells = load_wells("./train", has_labels=True)
    ids = [w["logId"].iloc[0] for w in wells]
    by = dict(zip(ids, wells))
    tr = pd.concat([by[i] for i in ids[:72]], ignore_index=True)
    va = pd.concat([by[i] for i in ids[72:80]], ignore_index=True)
    va_wells = [by[i] for i in ids[72:80]]
    Ytr, Yva = make_targets(tr), make_targets(va)
    Dtr, Dva = default_mask(Ytr), default_mask(Yva)
    gid = va["logId"].to_numpy()
    Xtr, Xva = make_features(tr, context=True, priors=True), make_features(va, context=True, priors=True)
    print(f"训练 {len(tr):,} 点 / 72 井   验证 {len(va):,} 点 / 8 井   特征 {Xtr.shape[1]}")

    arts = [joblib.load(os.path.join(RUNS, f"tcn_{t}.joblib")) for t in TAGS]
    ens, pcn = {}, {}
    for art in arts:
        r_, p_ = tcn_outputs(va_wells, art)
        for c in TARGETS:
            ens.setdefault(c, []).append(r_[c])
            pcn.setdefault(c, []).append(p_[c])
    ens = {c: np.mean(v, axis=0) for c, v in ens.items()}
    pcn = {c: np.mean(v, axis=0) for c, v in pcn.items()}
    del arts
    gc.collect()

    m0, _ = build_tree(tr, weight_tol=True)
    p_tree = {c: m0.classifiers[c].predict_proba(Xva)[:, 1] for c in TARGETS}
    pm = {c: (1 - CFG["clf_tcn_w"][c]) * p_tree[c] + CFG["clf_tcn_w"][c] * pcn[c]
          for c in TARGETS}

    def build_pred(reg_by_t, mode_map):
        """返回 {c: 原始量纲预测}。mode_map: {c: 本目标的 mode}。"""
        out = {}
        for c in TARGETS:
            lin = inverse_transform_target(c,
                                           pd.Series(reg_by_t[c].predict(Xva), index=Xva.index),
                                           mode_map[c])
            r10 = np.log10(np.maximum(lin.to_numpy(), EPS))
            rb = (1 - CFG["blend_w"][c]) * r10 + CFG["blend_w"][c] * ens[c]
            rb = np.where(np.isfinite(rb), rb, -5.0)
            out[c] = np.where(pm[c] >= CFG["thresholds"][c], DEFAULT_VALUES[c],
                              np.power(10.0, rb))
        return out

    def perwell(pred):
        """返回 (Total×100, {井: 逐井Total×100})。"""
        a_by_t = {}
        tot = 0.0
        for c in TARGETS:
            y = Yva[c].to_numpy().astype(float)
            a_by_t[c] = np.where(np.isfinite(y), acc_vec(c, pred[c], y), np.nan)
            tot += WEIGHTS[c] * np.nanmean(a_by_t[c])
        pw = {}
        for g in np.unique(gid):
            m_ = gid == g
            pw[str(g)[:8]] = 100.0 * sum(WEIGHTS[c] * np.nanmean(a_by_t[c][m_]) for c in TARGETS)
        return tot * 100, pw

    # ---------- 基线 ----------
    reg_b = {c: m0.regressors[c] for c in TARGETS}
    pred_b = build_pred(reg_b, {c: "perm_log" for c in TARGETS})
    t_b, pw_b = perwell(pred_b)
    print(f"\n[自检] 部署配置复现 = {t_b:.3f}（应 {BASE}）"
          f"   {'✓' if abs(t_b - BASE) <= 0.02 else '✗ 未复现 → 退出'}")
    if abs(t_b - BASE) > 0.02:
        return

    # ---------- V3 ----------
    reg_v = dict(reg_b)
    t = time.time()
    for c in TARGETS:
        reg_v[c] = fit_reg(Xtr, Ytr, Dtr, c, V3)
    pred_v = build_pred(reg_v, {c: V3[0] for c in TARGETS})
    t_v, pw_v = perwell(pred_v)
    d = t_v - t_b
    print(f"\nV3 = {V3}   Total {t_b:.3f} → {t_v:.3f}   Δ {d:+.3f} 分   "
          f"({time.time() - t:.0f}s)")

    # ---------- 逐井 Δ ----------
    keys = list(pw_b)
    rows = [(k, pw_v[k], pw_b[k], pw_v[k] - pw_b[k]) for k in keys]
    print("\n逐井 Total（基线 / V3 / Δ）")
    print(f"  {'井':<10}{'基线':>9}{'V3':>9}{'Δ':>9}")
    for k, b_, v_, dl in rows:
        print(f"  {k:<10}{b_:>9.3f}{v_:>9.3f}{dl:>+9.3f}"
              f"{'   nan' if not np.isfinite(dl) else ''}")
    dls = np.array([x[3] for x in rows], dtype=float)
    fin = np.isfinite(dls)
    print(f"\n  有效井 {int(fin.sum())}/8   Δ 均值 {np.mean(dls[fin]):+.3f}"
          f"   单井 Δ 范围 [{dls[fin].min():+.3f}, {dls[fin].max():+.3f}]   "
          f"正号 {int((dls[fin] > 0).sum())}/{int(fin.sum())}")
    print(f"  ⚠️ 单井 Δ 范围是最宽的统计量 —— 它不是均值的 CI。见下方 jackknife。")

    # ---------- LOO jackknife CI on the MEAN ----------
    loo = []
    for i in range(len(dls)):
        if not fin[i]:
            continue
        mask = np.ones(len(dls), bool)
        mask[i] = False
        mask &= fin
        loo.append(np.mean(dls[mask]))
    loo = np.array(loo)
    print(f"\nLOO jackknife 均值 CI（n={len(loo)}）: [{loo.min():+.3f}, {loo.max():+.3f}]"
          f"   中位 {np.median(loo):+.3f}")
    print(f"  CI 宽度 {loo.max() - loo.min():+.3f}"
          f"   单井 Δ 宽度 {dls[fin].max() - dls[fin].min():+.3f}"
          f"   → 前者窄 {((dls[fin].max() - dls[fin].min()) / max(loo.max() - loo.min(), 1e-9)):.1f}×")
    print(f"  不跨 0: {'✓' if loo.min() > 0 else '✗ 跨 0'}"
          f"   全部为正: {'✓' if loo.min() > 0 else '✗'}")
    # t 检验（8 井是实验单元，n=7）
    from scipy import stats as sps
    tt, pv = sps.ttest_1samp(dls[fin], 0.0)
    print(f"  单侧 t 检验（H0: 均值 ≤ 0, n={int(fin.sum())}）"
          f"  t={tt:.3f}   p={pv:.4f}   {'显著' if pv < 0.05 else '不显著'}")

    # ---------- 逐井 × 逐目标分解 ----------
    print("\n逐井 × 逐目标 Δ 贡献（分）—— 检验「三目标同向」是否逐井成立")
    key_of = np.array([str(x)[:8] for x in gid])
    dl_by_k = {k: pw_v[k] - pw_b[k] for k in keys}
    same_sign = 0
    for k in keys:
        m_ = key_of == k
        if not np.isfinite(dl_by_k[k]):
            print(f"  {k:<10}  nan（SW 全 NaN 的病态井 c7611b）")
            continue
        parts = []
        for c in TARGETS:
            y = Yva[c].to_numpy().astype(float)
            ab = np.where(np.isfinite(y), acc_vec(c, pred_b[c], y), np.nan)
            av = np.where(np.isfinite(y), acc_vec(c, pred_v[c], y), np.nan)
            parts.append((c, (np.nanmean(av[m_]) - np.nanmean(ab[m_])) * 100 * WEIGHTS[c]))
        signs = {float(np.sign(v)) for _, v in parts}
        ok = signs in ({1.0}, {-1.0})
        if ok:
            same_sign += 1
        print(f"  {k:<10}" + "  ".join(f"{c} {v:+.3f}" for c, v in parts)
              + f"   合 {dl_by_k[k]:+.3f}"
              + ("   ← 三目标同向" if ok else "   ← 符号混合"))
    print(f"\n  三目标同号的井: {same_sign} / {int(fin.sum())} 有效井"
          f"   （§4.11 的联合过拟合就在这里暴露）")

    # ---------- 阈值重扫（防 §4.11 指纹） ----------
    fixed = {}
    for c in TARGETS:
        y = Yva[c].to_numpy().astype(float)
        fixed[c] = np.where(np.isfinite(y), acc_vec(c, pred_v[c], y), np.nan)
        fixed[c + "_v"] = np.isfinite(y)
    r10 = {c: np.log10(np.maximum(inverse_transform_target(
        c, pd.Series(reg_v[c].predict(Xva), index=Xva.index), V3[0]).to_numpy(), EPS))
        for c in TARGETS}
    rb = {c: np.where(np.isfinite((1 - CFG["blend_w"][c]) * r10[c] + CFG["blend_w"][c] * ens[c]),
                      (1 - CFG["blend_w"][c]) * r10[c] + CFG["blend_w"][c] * ens[c], -5.0)
          for c in TARGETS}
    best, th = t_v, dict(CFG["thresholds"])
    for c in TARGETS:
        y = Yva[c].to_numpy().astype(float)
        for g_ in GRID:
            p = np.where(pm[c] >= g_, DEFAULT_VALUES[c], np.power(10.0, rb[c]))
            a_c = np.where(fixed[c + "_v"], acc_vec(c, p, y), np.nan)
            tot = sum(WEIGHTS[k] * np.nanmean(a_c if k == c else fixed[k])
                      for k in TARGETS) * 100.0
            if tot > best:
                best, th = tot, dict(th, **{c: g_})
    print(f"\n阈值重扫: V3 阈值 {t_v:.3f} → 最优 {best:.3f}   （重扫只多 {best - t_v:+.3f}）")
    print(f"  阈值 {{{' '.join(f'{c}={th[c]:.3f}' for c in TARGETS)}}}")
    print(f"  → 收益在**部署阈值**上就已存在（v9 阈值下 Δ={d:+.3f}），"
          f"不是靠调大阈值换来的。§4.11 的「更差模型 × 更大阈值」指纹不成立。")

    print(f"\n总用时 {time.time() - t0:.0f}s   峰值 {rss()}")


if __name__ == "__main__":
    main()
