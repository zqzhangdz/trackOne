"""det81 / locminmax101 的完整口径复核（tunex_nlinfeat.py 阶段 2 崩溃后的补跑）。

阶段 2 在计算逐井 Δ 时崩了（`base_sc[well_id]` 误当成 dict key，应为 Series 下标），
但 det81 的 clf300 AUC 已经打出来：
  POR  0.8966 (Δ vs 自检 -0.0000)   ← 快筛 clf60 的 +0.0044 **没有复现**
  PERM 0.8921 (Δ +0.0015)
  SW   0.9730 (Δ +0.0029)
所以快筛的 ΔAUC 排序不可信，必须用 clf300 重算全部量。本脚本只跑这两个候选，
且每次 fit 完立刻落盘 npz，崩了也能续。

基线用 `runs/screen_tree_cache.joblib`（与部署 v7 同源的树 p_def/reg，
`ctx_width_jackknife.py` 已验证其 v7 阈值下逐井合计 = 73.749），省掉 245 s×3 的重拟合。
另算两种对照，避免"候选换阈值、基线锁 v7 阈值"的口径不公：
  A) 双方都锁 v7 阈值  —— 纯分类器效应（§4.11 的教训：分类器必须在自己原有的阈值上评）
  B) 双方各自在验证集扫最优阈值 —— 部署口径的公平对照

用法: /opt/anaconda3/bin/python experiments/tunex_nlinfeat_verify.py
"""
from __future__ import annotations

import gc
import json
import os
import resource
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src.feature as F  # noqa: E402
from src.feature import (  # noqa: E402
    CURVE_COLS,
    default_mask,
    load_wells,
    make_targets,
    transform_target,
)
from src.model import POR_MONOTONIC, _clf, _reg  # noqa: E402
from src.utils import DEFAULT_VALUES, EPS, TARGETS, WEIGHTS  # noqa: E402

RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
N_TRAIN_WELLS, BND = 72, 657489
RAW13 = ["DEPTH", "GR", "PE", "SP", "CAL", "AC", "DEN", "CNL",
         "RXO", "RT", "DEVI", "AZIM", "BIT"]
LAGS = [-10, -5, -3, -2, -1, 1, 2, 3, 5, 10]
WINS = [5, 11, 21]
V7_TH = {"POR": 0.13, "PERM": 0.27, "SW": 0.05}
V7 = {"blend_w": {"POR": 0.50, "PERM": 0.50, "SW": 0.75},
      "thresholds": V7_TH, "clf_tcn_w": {"POR": 0.60, "PERM": 0.70, "SW": 0.30}}
TH_GRID = np.arange(0.01, 0.991, 0.01)
T0 = time.time()
PEAK = [0.0]


def rss_mb() -> float:
    v = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    if v > PEAK[0]:
        PEAK[0] = v
    return PEAK[0]


def say(s: str) -> None:
    print(s, flush=True)


def acc_vec(c, pred, true):
    p = np.asarray(pred, np.float64)
    t = np.asarray(true, np.float64)
    if c == "PERM":
        return np.maximum(0.0, 1.0 - np.abs(np.log10(np.maximum(p / np.maximum(t, EPS), EPS))))
    tol = 0.08 if c == "POR" else 0.05
    return np.maximum(0.0, 1.0 - np.abs(p - t) / (tol * (np.abs(t) + EPS)))


def log10ify(c, r):
    r = np.asarray(r, np.float64)
    return r if c == "PERM" else np.log10(np.maximum(r, EPS))


def acc_at(pm, c, rt, reg_tcn, pdf_tcn, y, m, th, cw, bw):
    pmv = (1 - cw) * pm + cw * pdf_tcn
    regb = np.power(10.0, (1 - bw) * rt + bw * reg_tcn)
    return float(acc_vec(c, np.where(pmv >= th, DEFAULT_VALUES[c], regb)[m], y[m]).mean())


def sweep_best(pm, c, rt, reg_tcn, pdf_tcn, y, m, cw, bw):
    pmv = ((1 - cw) * pm + cw * pdf_tcn)[m]
    regb = np.power(10.0, (1 - bw) * rt + bw * reg_tcn)[m]
    fd = acc_vec(c, np.full(int(m.sum()), DEFAULT_VALUES[c]), y[m])
    fr = acc_vec(c, regb, y[m])
    order = np.argsort(pmv, kind="stable")
    pos = np.searchsorted(pmv[order], TH_GRID, side="left")
    Fd = np.concatenate([[0.0], np.cumsum(fd[order])])
    Fr = np.concatenate([[0.0], np.cumsum(fr[order])])
    n = len(pmv)
    a = (Fr[pos] + (Fd[n] - Fd[pos])) / n
    j = int(np.argmax(a))
    return float(a[j]), float(TH_GRID[j])


def well_scores(pm, c, rt, reg_tcn, pdf_tcn, y, m, th, cw, bw, well_ids):
    """逐井得分贡献（分）；三目标相加 == 官方 Total。必须带 ×100。"""
    pmv = (1 - cw) * pm + cw * pdf_tcn
    regb = np.power(10.0, (1 - bw) * rt + bw * reg_tcn)
    row = np.nan_to_num(100.0 * WEIGHTS[c] * acc_vec(c, np.where(pmv >= th, DEFAULT_VALUES[c], regb), y))
    nv = m.astype(int)
    return pd.Series(np.asarray(row, np.float64) * nv / float(nv.sum()),
                     index=well_ids).groupby(level=0).sum()


def det_feat(raw, g, w):
    return pd.DataFrame({f"det{w}_{c}": raw[c].groupby(g, group_keys=False).apply(
        lambda s: s - s.rolling(w, center=True, min_periods=1).median()) for c in CURVE_COLS})


def mm_feat(raw, g, w):
    out = {}
    for c in CURVE_COLS:
        s = raw[c]
        mn = s.groupby(g, group_keys=False).transform(lambda s: s.rolling(w, center=True, min_periods=1).min())
        mx = s.groupby(g, group_keys=False).transform(lambda s: s.rolling(w, center=True, min_periods=1).max())
        out[f"lo{w}_{c}"] = (s - mn).to_numpy()
        out[f"hi{w}_{c}"] = (mx - s).to_numpy()
    return pd.DataFrame(out)


def auc_loo_well(prob, lab, well_ids):
    v = {}
    for w in dict.fromkeys(well_ids):
        iw = well_ids == w
        v[w] = float(roc_auc_score(lab[iw], prob[iw]))
    jk = {w: float(np.mean([v[x] for x in v if x != w])) for w in v}
    return min(jk.values()), max(jk.values()), v


def fit_target(Xtr, Xva, Ytr, Dtr, ytr, c):
    v = ~np.isnan(ytr[c])
    clf = _clf()
    clf.fit(Xtr.loc[v], Dtr.loc[v, c].astype(int).to_numpy())
    p = clf.predict_proba(Xva)[:, 1].astype(np.float32)
    del clf
    gc.collect()
    real = v & (~Dtr[c].to_numpy().astype(bool))
    reg = _reg(monotonic=POR_MONOTONIC if c == "POR" else None, columns=list(Xtr.columns))
    yt = transform_target(c, Ytr[c].loc[real]).to_numpy()
    yraw = Ytr[c].loc[real].to_numpy().astype(np.float64)
    w = np.abs(np.log10(np.maximum(yraw, EPS))) if c == "PERM" else np.abs(yraw)
    w = np.clip(w, np.percentile(w, 5), np.percentile(w, 95))
    w = (w / w.mean()).astype(np.float32)
    reg.fit(Xtr.loc[real], yt, sample_weight=w)
    r = log10ify(c, reg.predict(Xva)).astype(np.float32)
    del reg
    gc.collect()
    return p, r


def main() -> None:
    F.LAG_OFFSETS, F.WIN_SIZES = list(LAGS), list(WINS)
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
    m_va = {c: ~np.isnan(yva[c]) for c in TARGETS}
    Z = np.load(os.path.join(RUNS, "tcn_all80_parts.npz"))
    ens = {c: np.float32(Z[f"reg_{c}"][BND:]) for c in TARGETS}
    pdf = {c: np.float32(Z[f"pdf_{c}"][BND:]) for c in TARGETS}

    Xtr = F.make_features(tr, context=True, priors=True)
    Xva = F.make_features(va, context=True, priors=True)
    assert Xtr.shape[1] == 225, Xtr.shape
    say("=" * 100)
    say("det81 / locminmax101 完整口径复核（clf300 + weight_tol 回归；基线 = screen_tree_cache 部署同源）")
    say("=" * 100)

    # ── 基线：缓存（部署同源），两种阈值
    st = joblib.load(os.path.join(RUNS, "screen_tree_cache.joblib"))
    say(f"缓存基线 {len(st['p_def']['POR'])} 点；AUC 复核 "
        + " ".join(f"{c} {roc_auc_score(Dva[c].to_numpy().astype(int)[m_va[c]], st['p_def'][c][m_va[c]]):.4f}"
                   for c in TARGETS))
    base = {"v7": {}, "best": {}}
    base_th_best = {}
    for c in TARGETS:
        base["v7"][c] = well_scores(st["p_def"][c], c, np.asarray(st["reg"][c], np.float64),
                                    ens[c], pdf[c], yva[c], m_va[c], V7_TH[c],
                                    V7["clf_tcn_w"][c], V7["blend_w"][c], yid)
        a, th = sweep_best(st["p_def"][c], c, np.asarray(st["reg"][c], np.float64),
                           ens[c], pdf[c], yva[c], m_va[c], V7["clf_tcn_w"][c], V7["blend_w"][c])
        base_th_best[c] = th
        base["best"][c] = well_scores(st["p_def"][c], c, np.asarray(st["reg"][c], np.float64),
                                      ens[c], pdf[c], yva[c], m_va[c], th,
                                      V7["clf_tcn_w"][c], V7["blend_w"][c], yid)
    say(f"基线：v7 阈值合计 {float(sum(base['v7'].values()).sum()):.3f}"
        f"   各自最优阈值合计 {float(sum(base['best'].values()).sum()):.3f}"
        f"   (最优 th " + " ".join(f"{c} {base_th_best[c]:.3f}" for c in TARGETS) + ")")

    out = {"base_total_v7": float(sum(base["v7"].values()).sum()),
           "base_total_best": float(sum(base["best"].values()).sum()),
           "base_th_best": base_th_best, "candidates": {}}

    for name, mk in [("det81", lambda raw, g: det_feat(raw, g, 81)),
                     ("locminmax101", lambda raw, g: mm_feat(raw, g, 101))]:
        say("\n" + "─" * 100)
        say(f"候选 {name}")
        say("─" * 100)
        Xtr_c = pd.concat([Xtr, mk(Xtr[RAW13].copy(), tr["logId"].to_numpy())], axis=1)
        Xva_c = pd.concat([Xva, mk(Xva[RAW13].copy(), va["logId"].to_numpy())], axis=1)
        say(f"  {Xtr_c.shape[1]} 特征  [{time.time()-T0:.0f}s]")
        npz = os.path.join(RUNS, f"nlinfeat_{name}.npz")
        if os.path.exists(npz) and len(np.load(npz).files) >= 2 * len(TARGETS):
            d = np.load(npz)
            cp = {c: d[f"p_{c}"] for c in TARGETS}
            cr = {c: d[f"r_{c}"] for c in TARGETS}
            say(f"  从缓存 {os.path.basename(npz)} 载入，跳过拟合")
        else:
            cp, cr = {}, {}
            for c in TARGETS:
                p, r = fit_target(Xtr_c, Xva_c, Ytr, Dtr, ytr, c)
                cp[c], cr[c] = p, r
                np.savez(npz, **{f"p_{c}": cp[c], f"r_{c}": cr[c]})
                say(f"  {c}: 拟合+落盘完成  [{time.time()-T0:.0f}s]  RSS {rss_mb():.0f} MB")
            del Xtr_c, Xva_c
            gc.collect()
        rec = {}
        for thset, ths in (("v7", V7_TH), ("best", {})):
            tot, sc = 0.0, {}
            for c in TARGETS:
                if thset == "v7":
                    th = V7_TH[c]
                else:
                    a, th = sweep_best(cp[c], c, cr[c], ens[c], pdf[c], yva[c], m_va[c],
                                       V7["clf_tcn_w"][c], V7["blend_w"][c])
                    rec[f"acc_best_{c}"] = a
                    rec[f"th_best_{c}"] = th
                sc[c] = well_scores(cp[c], c, cr[c], ens[c], pdf[c], yva[c], m_va[c], th,
                                    V7["clf_tcn_w"][c], V7["blend_w"][c], yid)
                tot += float(sc[c].sum())
                if thset == "v7":
                    rec[f"acc_v7th_{c}"] = float(acc_at(cp[c], c, cr[c], ens[c], pdf[c], yva[c], m_va[c],
                                                        th, V7["clf_tcn_w"][c], V7["blend_w"][c]))
            rec[f"total_{thset}"] = tot
            rec[f"delta_{thset}"] = tot - out[f"base_total_{thset}"]
            totb = sum(base[thset].values())
            dw = {k: float(sum(sc[k] for sc in [base[thset][c] for c in TARGETS])
                           if False else (sum(sc[c][k] for c in TARGETS)
                                          - sum(base[thset][c][k] for c in TARGETS)))
                  for k in yid_dict_keys(yid)}
            d = rec[f"delta_{thset}"]
            rec[f"loo_{thset}"] = [d - max(dw.values()), d - min(dw.values())]
            rec[f"wells_pos_{thset}"] = sum(1 for v in dw.values() if v > 0)
            rec[f"per_well_{thset}"] = dw
        for c in TARGETS:
            rec[f"auc_{c}"] = float(roc_auc_score(Dva[c].to_numpy().astype(int)[m_va[c]], cp[c][m_va[c]]))
            lo, hi, perw = auc_loo_well(cp[c], Dva[c].to_numpy().astype(int), yid)
            rec[f"auc_loo_{c}"] = [lo, hi]
        out["candidates"][name] = rec
        say(f"  AUC  POR {rec['auc_POR']:.4f} (基线 0.8966, Δ {rec['auc_POR']-0.8966:+.4f}) | "
            f"PERM {rec['auc_PERM']:.4f} (0.8906, Δ {rec['auc_PERM']-0.8906:+.4f}) | "
            f"SW {rec['auc_SW']:.4f} (0.9701, Δ {rec['auc_SW']-0.9701:+.4f})")
        say(f"  Total@v7阈值 {rec['total_v7']:.3f} (Δ {rec['delta_v7']:+.3f}) | "
            f"Total@最优阈值 {rec['total_best']:.3f} (Δ {rec['delta_best']:+.3f})")
        say(f"  acc@v7阈值  POR {rec['acc_v7th_POR']:.4f} / PERM {rec['acc_v7th_PERM']:.4f} / SW {rec['acc_v7th_SW']:.4f}")
        say(f"  acc@最优阈值 POR {rec['acc_best_POR']:.4f} / PERM {rec['acc_best_PERM']:.4f} / SW {rec['acc_best_SW']:.4f}")
        say(f"  留一井 jackknife  v7口径 {rec['loo_v7'][0]:+.3f}..{rec['loo_v7'][1]:+.3f}"
            f"({rec['wells_pos_v7']}/8 正)   最优阈值口径 {rec['loo_best'][0]:+.3f}..{rec['loo_best'][1]:+.3f}"
            f"({rec['wells_pos_best']}/8 正)")
        say(f"  AUC 留一井      POR {rec['auc_loo_POR'][0]:.4f}..{rec['auc_loo_POR'][1]:.4f} | "
            f"PERM {rec['auc_loo_PERM'][0]:.4f}..{rec['auc_loo_PERM'][1]:.4f} | "
            f"SW {rec['auc_loo_SW'][0]:.4f}..{rec['auc_loo_SW'][1]:.4f}")
        say(f"  [{time.time()-T0:.0f}s]  RSS {rss_mb():.0f} MB")
        with open(os.path.join(RUNS, "nlinfeat_verify.json"), "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)

    say(f"\n峰值 RSS {rss_mb():.0f} MB；总耗时 {time.time()-T0:.0f}s")


def yid_dict_keys(yid):
    return list(dict.fromkeys(yid))


if __name__ == "__main__":
    main()
