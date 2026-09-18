"""用缓存前向做 LOO：7 井调 (w_tcn, clf_w, threshold) -> 留出井评估，逐目标独立调优。

报告各候选族相对 A(=T10+T11, 120ep 部署同源) 的 LOO 增益与配对 bootstrap。
纯 numpy，不 import torch/sklearn fit。
"""
from __future__ import annotations
import os, sys, warnings
warnings.filterwarnings("ignore")
import joblib, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.feature import default_mask, load_wells, make_targets
from src.utils import DEFAULT_VALUES, EPS, TARGETS, TOLERANCE, WEIGHTS

RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")


def pw(c, pred, true):
    p = np.asarray(pred, float); t = np.asarray(true, float)
    if c == "PERM":
        p = np.log10(np.maximum(p, EPS)); t = np.log10(np.maximum(t, EPS))
    return np.maximum(0.0, 1.0 - np.abs(p - t) / (TOLERANCE[c] * (np.abs(t) + EPS)))


def main():
    wells = load_wells("./train", has_labels=True); ids = [w["logId"].iloc[0] for w in wells]
    va = [w for w in wells if w["logId"].iloc[0] in set(ids[72:])]
    va_df = pd.concat(va, ignore_index=True); Y = make_targets(va_df); td = default_mask(Y)
    larr = va_df["logId"].to_numpy(); uniq = list(pd.unique(larr))
    cache = joblib.load(os.path.join(RUNS, "screen_tree_cache.joblib")); rt, pt = cache["reg"], cache["p_def"]
    store = {k: v for k, v in np.load(os.path.join(RUNS, "tcn_screen_outputs.npz")).items()}

    fams = {
        "A  T10+T11 (120x2)": ["T10_ln10", "T11_seed43_ln10"],
        "A3 +T14 (120x3)": ["T10_ln10", "T11_seed43_ln10", "T14_seed44_ln10"],
        "D  T12+T10+T11": ["T12_seed42_200ep", "T10_ln10", "T11_seed43_ln10"],
        "H  T12+T13+T11+T14": ["T12_seed42_200ep", "T13_seed43_200ep", "T11_seed43_ln10", "T14_seed44_ln10"],
        "E  T12+T10+T14": ["T12_seed42_200ep", "T10_ln10", "T14_seed44_ln10"],
    }
    ths = np.arange(0.02, 0.96, 0.02); wg = np.arange(0.0, 1.01, 0.05); cwg = np.arange(0.0, 1.01, 0.25)

    def score(tags, mask):
        tot = 0.0
        for c in TARGETS:
            ens = np.mean([store[f"{t}__{c}__reg"] for t in tags], axis=0)
            pcn = np.mean([store[f"{t}__{c}__pdf"] for t in tags], axis=0)
            y = Y[c].to_numpy(); m = (~np.isnan(y)) & mask
            yy = y[m]
            f_def = pw(c, np.full_like(yy, DEFAULT_VALUES[c]), yy)
            rt_m, pt_m = rt[c][m], pt[c][m]
            best = -1.0
            for cw in cwg:
                pm = (1 - cw) * pt_m + cw * pcn[m]
                order = np.argsort(pm, kind="stable"); pm_s = pm[order]
                cs_r0 = np.concatenate([[0.0], np.cumsum(f_def[order])])
                N = len(yy)
                for w in wg:
                    f_real = pw(c, np.power(10., (1 - w) * rt_m + w * ens[m]), yy)
                    cs_r = np.concatenate([[0.0], np.cumsum(f_real[order])])
                    idx = np.searchsorted(pm_s, ths, side="left")
                    acc = (cs_r[idx] + (cs_r0[N] - cs_r0[idx])) / N
                    j = int(np.argmax(acc))
                    if acc[j] > best: best = float(acc[j])
            tot += WEIGHTS[c] * best
        return tot * 100

    full = np.ones(len(larr), bool)
    print("=== LOO（7 井调参 -> 留出井）===")
    loo = {}
    for name, tags in fams.items():
        vals = []
        for hold in uniq:
            ho = larr == hold
            # 在其余 7 井调参 -> 留出井评估（逐目标独立）
            vals.append(_loo_eval(store, rt, pt, Y, td, larr, tags, ho, ths, wg, cwg))
        loo[name] = np.array(vals)
        print(f"  {name:22s} mean={np.nanmean(vals):.3f} per-well=" + " ".join(f"{v:.1f}" for v in vals))


def _loo_eval(store, rt, pt, Y, td, larr, tags, hold_mask, ths, wg, cwg, _c=None):
    """在 ~hold_mask 上调参，在 hold_mask 上评估（逐目标独立）。"""
    tot = 0.0
    for c in TARGETS:
        ens = np.mean([store[f"{t}__{c}__reg"] for t in tags], axis=0)
        pcn = np.mean([store[f"{t}__{c}__pdf"] for t in tags], axis=0)
        y = Y[c].to_numpy()
        base_m = ~np.isnan(y)
        tr = base_m & ~hold_mask
        ho = base_m & hold_mask
        if ho.sum() == 0: return np.nan
        yy_tr = y[tr]; f_def_tr = pw(c, np.full_like(yy_tr, DEFAULT_VALUES[c]), yy_tr)
        rt_tr, pt_tr = rt[c][tr], pt[c][tr]
        best = (-1.0, 0.7, 0.10, 0.5)
        for cw in cwg:
            pm = (1 - cw) * pt_tr + cw * pcn[tr]
            order = np.argsort(pm, kind="stable"); N = len(yy_tr)
            cs0 = np.concatenate([[0.0], np.cumsum(f_def_tr[order])])
            for w in wg:
                fr = pw(c, np.power(10., (1 - w) * rt_tr + w * ens[tr]), yy_tr)
                cs = np.concatenate([[0.0], np.cumsum(fr[order])])
                for t in ths:
                    pred = np.where(pm >= t, DEFAULT_VALUES[c],
                                    np.power(10., (1 - w) * rt_tr + w * ens[tr]))
                    a = pw(c, pred, yy_tr).mean()
                    if a > best[0]: best = (a, w, t, cw)
        _, w, t, cw = best
        pm_ho = (1 - cw) * pt[c][ho] + cw * pcn[ho]
        pred_ho = np.where(pm_ho >= t, DEFAULT_VALUES[c],
                           np.power(10., (1 - w) * rt[c][ho] + w * ens[ho]))
        tot += WEIGHTS[c] * pw(c, pred_ho, y[ho]).mean()
    return tot * 100


if __name__ == "__main__":
    main()
