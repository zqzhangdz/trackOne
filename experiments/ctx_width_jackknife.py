"""±60/w41 上下文宽度的留一井复核（§6：|Δ| ≥ +0.05 必须复核，不能只看单切分）。

背景：`tune_ctx_width.py` 报 ±60/w41 → 73.832（Δ +0.058，基线 ±10/w21 = 73.774），
字面过了 +0.05 晋级线，但落在 ±0.3 噪声带内，且 AUC 在 POR/PERM 上**下降**
（0.8966→0.8927、0.8906→0.8867），只有 SW 上升（0.9701→0.9730，已近上限）。
单切分不能判，必须按井复核 —— 8 口井才是实验单元（行级 bootstrap 在本项目无效，
单井内 POR lag-1 自相关 0.967）。

做法：只跑 ±60/w41 一个配置（约 4 min），逐井算总分；基线直接用
`runs/screen_tree_cache.joblib`（与部署 v7 同源的树 p_def/reg）逐井算，两者相减得
逐井 Δ，再做留一 jackknife。纯 sklearn，不 import torch，不跑 TCN。

用法: python experiments/ctx_width_jackknife.py
"""
from __future__ import annotations
import gc, os, sys, time, warnings
warnings.filterwarnings("ignore")
import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src.feature as F
from src.feature import default_mask, load_wells, make_targets, transform_target
from src.model import POR_MONOTONIC, _clf, _reg
from src.utils import DEFAULT_VALUES, EPS, TARGETS, WEIGHTS

RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
N_TRAIN_WELLS = 72
# 与 tune_ctx_width.py 的最优配置一致
LAGS = [-60, -30, -20, -10, -5, -3, -2, -1, 1, 2, 3, 5, 10, 20, 30, 60]
WINS = [5, 11, 21, 41]
V7 = {"blend_w": {"POR": 0.50, "PERM": 0.50, "SW": 0.75},
      "thresholds": {"POR": 0.13, "PERM": 0.27, "SW": 0.05},
      "clf_tcn_w": {"POR": 0.60, "PERM": 0.70, "SW": 0.30}}
TH_GRID = np.arange(0.01, 0.991, 0.01)
T0 = time.time()


def acc_vec(c, pred, true):
    p = np.asarray(pred, np.float64); t = np.asarray(true, np.float64)
    if c == "PERM":
        return np.maximum(0.0, 1.0 - np.abs(np.log10(np.maximum(p / np.maximum(t, EPS), EPS))))
    tol = 0.08 if c == "POR" else 0.05
    return np.maximum(0.0, 1.0 - np.abs(p - t) / (tol * (np.abs(t) + EPS)))


def log10ify(c, r):
    r = np.asarray(r, np.float64)
    return r if c == "PERM" else np.log10(np.maximum(r, EPS))


def sweep_best(pm_tree, c, rt_log, reg_tcn, pdf_tcn, y, m, cw, bw):
    """在验证集上扫阈值，返回 (最优 acc, 最优 th)。cw 用于概率融合，bw 用于回归融合。"""
    pmv = ((1 - cw) * pm_tree + cw * pdf_tcn)[m]
    regb = np.power(10.0, (1 - bw) * rt_log + bw * reg_tcn)[m]
    fd = acc_vec(c, np.full(int(m.sum()), DEFAULT_VALUES[c]), y[m])
    fr = acc_vec(c, regb, y[m])
    order = np.argsort(pmv, kind="stable")
    pos = np.searchsorted(pmv[order], TH_GRID, side="left")
    Fd = np.concatenate([[0.0], np.cumsum(fd[order])])
    Fr = np.concatenate([[0.0], np.cumsum(fr[order])])
    n = len(pmv)
    acc = (Fr[pos] + (Fd[n] - Fd[pos])) / n
    j = int(np.argmax(acc))
    return float(acc[j]), float(TH_GRID[j])


def well_scores(pm_tree, c, rt_log, reg_tcn, pdf_tcn, y, m, th, cw, bw, well_ids):
    """逐井**得分贡献**（单位：分）。三目标相加 == 官方 Total。

    Total = 100·Σ_c WEIGHTS[c]·mean_c(acc)，mean 是逐点等权。把每个点的贡献
    除以**全验证集的**有效点数再按井求和，既保持逐点等权，又可按井可加分解。
    注意必须带 ×100 —— 漏掉会让合计变成 0.737 而不是 73.749（踩过的坑）。
    """
    pmv = (1 - cw) * pm_tree + cw * pdf_tcn
    regb = np.power(10.0, (1 - bw) * rt_log + bw * reg_tcn)
    pred = np.where(pmv >= th, DEFAULT_VALUES[c], regb)
    row = np.nan_to_num(100.0 * WEIGHTS[c] * acc_vec(c, pred, y))  # 无效行 acc 为 NaN，置 0
    n_valid = m.astype(int)
    n_tot = float(n_valid.sum())
    s = np.asarray(row, np.float64) * n_valid / n_tot   # 逐点等权 → 逐井可加分解
    return pd.Series(s, index=well_ids).groupby(level=0).sum()


def main() -> None:
    wells = load_wells("./train", has_labels=True)
    ids = [w["logId"].iloc[0] for w in wells]
    va_ids = set(ids[N_TRAIN_WELLS:])
    tr = pd.concat([w for w in wells if w["logId"].iloc[0] not in va_ids], ignore_index=True)
    va = pd.concat([w for w in wells if w["logId"].iloc[0] in va_ids], ignore_index=True)
    yid = va["logId"].to_numpy()
    Ytr, Yva = make_targets(tr), make_targets(va)
    Dtr, Dva = default_mask(Ytr), default_mask(Yva)
    ytr = {c: Ytr[c].to_numpy() for c in TARGETS}
    yva = {c: Yva[c].to_numpy() for c in TARGETS}

    bnd = len(tr)
    Z = np.load(os.path.join(RUNS, "tcn_all80_parts.npz"))
    ens = {c: np.float32(Z[f"reg_{c}"][bnd:]) for c in TARGETS}
    pdf = {c: np.float32(Z[f"pdf_{c}"][bnd:]) for c in TARGETS}

    # ── 基线：screen_tree_cache（与部署 v7 同源的树 p_def/reg）
    st = joblib.load(os.path.join(RUNS, "screen_tree_cache.joblib"))
    base_th, base_scores = {}, {}
    for c in TARGETS:
        base_th[c] = V7["thresholds"][c]
        base_scores[c] = well_scores(st["p_def"][c], c, np.asarray(st["reg"][c], np.float64),
                                     ens[c], pdf[c], yva[c], ~np.isnan(yva[c]),
                                     base_th[c], V7["clf_tcn_w"][c], V7["blend_w"][c], yid)
    base_wells = sum(base_scores.values())

    # ── 自检：基线逐井贡献相加必须等于部署 v7 = 73.749（官方口径，逐点等权）
    dep = 73.749
    print(f"自检：基线逐井合计 {float(base_wells.sum()):.4f} vs 部署 v7 {dep} "
          f"（差 {float(base_wells.sum())-dep:+.5f}）→ "
          f"{'通过' if abs(float(base_wells.sum())-dep) < 5e-3 else '失败，不要相信下面的 Δ'}",
          flush=True)

    # ── 候选：±60/w41
    print(f"训练 ±60/w41（{len(LAGS)} lag × {len(WINS)} 窗口）...", flush=True)
    F.LAG_OFFSETS, F.WIN_SIZES = list(LAGS), list(WINS)
    Xtr = F.make_features(tr, context=True, priors=True)
    Xva = F.make_features(va, context=True, priors=True)
    print(f"  {Xtr.shape[1]} 特征  [{time.time()-T0:.0f}s]", flush=True)

    cand_th, cand_scores = {}, {}
    for c in TARGETS:
        v = ~np.isnan(ytr[c])
        clf = _clf()
        clf.fit(Xtr.loc[v], Dtr.loc[v, c].astype(int).to_numpy())
        p_va = clf.predict_proba(Xva)[:, 1].astype(np.float32)
        del clf; gc.collect()

        real = v & (~Dtr[c].to_numpy().astype(bool))
        reg = _reg(monotonic=POR_MONOTONIC if c == "POR" else None, columns=list(Xtr.columns))
        yt = transform_target(c, Ytr[c].loc[real]).to_numpy()
        yraw = Ytr[c].loc[real].to_numpy().astype(np.float64)
        w = np.abs(np.log10(np.maximum(yraw, EPS))) if c == "PERM" else np.abs(yraw)
        w = np.clip(w, np.percentile(w, 5), np.percentile(w, 95))
        w = (w / w.mean()).astype(np.float32)
        reg.fit(Xtr.loc[real], yt, sample_weight=w)
        rt_va = log10ify(c, reg.predict(Xva)).astype(np.float32)
        del reg; gc.collect()

        m = ~np.isnan(yva[c])
        a, th = sweep_best(p_va, c, rt_va, ens[c], pdf[c], yva[c], m,
                           V7["clf_tcn_w"][c], V7["blend_w"][c])
        cand_th[c] = th
        cand_scores[c] = well_scores(p_va, c, rt_va, ens[c], pdf[c], yva[c], m,
                                     th, V7["clf_tcn_w"][c], V7["blend_w"][c], yid)
        print(f"  {c}: 最优阈值 {th:.3f}（基线 {base_th[c]:.3f}）, "
              f"acc {a:.4f}  [{time.time()-T0:.0f}s]", flush=True)
    del Xtr, Xva
    gc.collect()

    cand_wells = sum(cand_scores.values())

    # ── 逐目标分解
    total_b, total_c = float(base_wells.sum()), float(cand_wells.sum())
    print("\n" + "=" * 90)
    print("逐目标分解（单位：分；候选 = ±60/w41，基线 = 部署 v7 ±10/w21）")
    print("=" * 90)
    print(f"{'目标':6s}{'权重':>6s}{'基线贡献':>11s}{'候选贡献':>11s}{'Δ':>10s}{'最优th':>9s}{'基线th':>9s}")
    for c in TARGETS:
        print(f"{c:6s}{WEIGHTS[c]:6.2f}{base_scores[c].sum():11.3f}{cand_scores[c].sum():11.3f}"
              f"{cand_scores[c].sum()-base_scores[c].sum():+10.3f}"
              f"{cand_th[c]:9.3f}{base_th[c]:9.3f}")
    print("-" * 90)

    # ── 逐井 Δ
    delta = {k: float(cand_wells[k] - base_wells[k]) for k in base_wells.index}
    print("\n" + "=" * 90)
    print(f"逐井分解（screen 8 井，8 井才是实验单元；候选 ±60/w41 vs 基线部署 v7）")
    print("=" * 90)
    print(f"{'井':34s}{'基线':>9s}{'候选':>9s}{'Δ总分':>10s}")
    for k in sorted(delta):
        print(f"{k:34s}{base_wells[k]:9.3f}{cand_wells[k]:9.3f}{delta[k]:+10.3f}")
    print("-" * 90)
    print(f"{'合计':34s}{total_b:9.3f}{total_c:9.3f}{total_c-total_b:+10.3f}")
    pos = sum(1 for v in delta.values() if v > 0)
    neg = sum(1 for v in delta.values() if v < 0)
    lo, hi = total_c - total_b - max(delta.values()), total_c - total_b - min(delta.values())
    d = total_c - total_b
    print(f"\n  正号井 {pos}/{len(delta)}，负号井 {neg}/{len(delta)}")
    print(f"  逐井 Δ 范围 [{min(delta.values()):+.3f}, {max(delta.values()):+.3f}]")
    print(f"  留一井 jackknife [{lo:+.3f}, {hi:+.3f}] → "
          f"{'跨 0：噪声内，不构成晋级证据' if lo <= 0 <= hi else '不跨 0：方向可信'}")
    print(f"  screen 噪声 ≈ ±0.3；|Δ| = {abs(d):.3f} → "
          f"{'超出噪声带' if abs(d) >= 0.3 else '在噪声带内'}")
    print(f"  晋级线 +0.05：字面 {'过' if d >= 0.05 else '不过'}；§6 要求同时过 +0.05 且方向一致")
    print(f"  耗时 {time.time()-T0:.1f}s")


if __name__ == "__main__":
    main()
