"""固定 v4 部署权重/阈值、只换工件族的干净对照（点加权口径，与 68.576 基线可比）。

固定：w_tcn = POR0.60/PERM0.95/SW0.75, th = 0.10/0.05/0.05, clf_w = 0.5。
A = T10+T11 (120ep) 为部署同源基线。另报逐井 total 差值（配对 bootstrap 95% CI）。
"""
from __future__ import annotations
import os, sys, warnings
warnings.filterwarnings("ignore")
import joblib, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.feature import default_mask, load_wells, make_features, make_targets
from src.utils import DEFAULT_VALUES, EPS, TARGETS, TOLERANCE, WEIGHTS

RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
CACHE = os.path.join(RUNS, "tcn_screen_outputs.npz")
V4W = {"POR": 0.60, "PERM": 0.95, "SW": 0.75}
V4T = {"POR": 0.10, "PERM": 0.05, "SW": 0.05}
CLFW = 0.5


def pw(c, pred, true):
    p = np.asarray(pred, float); t = np.asarray(true, float)
    if c == "PERM":
        p = np.log10(np.maximum(p, EPS)); t = np.log10(np.maximum(t, EPS))
    return np.maximum(0.0, 1.0 - np.abs(p - t) / (TOLERANCE[c] * (np.abs(t) + EPS)))


def main():
    wells = load_wells("./train", has_labels=True); ids = [w["logId"].iloc[0] for w in wells]
    va = [w for w in wells if w["logId"].iloc[0] in set(ids[72:])]
    va_df = pd.concat(va, ignore_index=True); Y = make_targets(va_df); tdef = default_mask(Y)
    larr = va_df["logId"].to_numpy(); uniq = list(pd.unique(larr))
    cache = joblib.load(os.path.join(RUNS, "screen_tree_cache.joblib")); reg_tree, p_tree = cache["reg"], cache["p_def"]
    store = {k: v for k, v in np.load(CACHE).items()}

    fams = {
        "A  T10+T11 (120+120)": ["T10_ln10", "T11_seed43_ln10"],
        "B  T12+T11 (200+120)": ["T12_seed42_200ep", "T11_seed43_ln10"],
        "C  T12+T10 (200+120)": ["T12_seed42_200ep", "T10_ln10"],
        "E  T12+T13 (200+200)": ["T12_seed42_200ep", "T13_seed43_200ep"],
        "F  T12+T13+T11 (200+200+120)": ["T12_seed42_200ep", "T13_seed43_200ep", "T11_seed43_ln10"],
        "G  all4": ["T10_ln10", "T11_seed43_ln10", "T12_seed42_200ep", "T13_seed43_200ep"],
    }

    def eval_fam(tags):
        """点加权 total + 逐目标 parts + 逐井 total 数组"""
        tot, parts = 0.0, {}
        per_well = np.zeros(len(uniq))
        for c in TARGETS:
            ens = np.mean([store[f"{t}__{c}__reg"] for t in tags], axis=0)
            pcn = np.mean([store[f"{t}__{c}__pdf"] for t in tags], axis=0)
            y = Y[c].to_numpy()
            pm = (1 - CLFW) * p_tree[c] + CLFW * pcn
            pred = np.where(pm >= V4T[c], DEFAULT_VALUES[c],
                            np.power(10., (1 - V4W[c]) * reg_tree[c] + V4W[c] * ens))
            m = ~np.isnan(y)
            f = pw(c, pred[m], y[m]); lw = larr[m]
            mdef = tdef[c].to_numpy()[m]; mr = ~mdef
            acc = f.mean()
            parts[c] = (acc, f[mr].mean() if mr.sum() else np.nan, f[mdef].mean() if mdef.sum() else np.nan)
            tot += WEIGHTS[c] * acc
            for i, wid in enumerate(uniq):
                sel = lw == wid
                if sel.sum(): per_well[i] += WEIGHTS[c] * f[sel].mean() * 100
        return tot * 100, parts, per_well

    results = {}
    print(f"{'family':30s} {'total':>7s}  real(POR,PERM,SW)           def(POR,PERM,SW)")
    for name, tags in fams.items():
        t, parts, pwd = eval_fam(tags)
        results[name] = pwd
        print(f"{name:30s} {t:7.3f}  ({parts['POR'][1]:.4f},{parts['PERM'][1]:.4f},{parts['SW'][1]:.4f})     "
              f"({parts['POR'][2]:.4f},{parts['PERM'][2]:.4f},{parts['SW'][2]:.4f})")

    A = results["A  T10+T11 (120+120)"]
    print("\n=== 相对 A 的逐井 total 差值（配对 bootstrap 95% CI）===")
    rng = np.random.default_rng(0)
    for name, pwv in results.items():
        if name.startswith("A"): continue
        d = pwv - A
        boot = np.array([np.nanmean(d[rng.integers(0, len(d), len(d))]) for _ in range(5000)])
        print(f"  {name:30s} d={np.nanmean(d):+.3f} boot95=[{np.nanpercentile(boot,2.5):+.3f},{np.nanpercentile(boot,97.5):+.3f}] "
              f"pos/neg={int(np.nansum(d>0))}/{int(np.nansum(d<0))}")


if __name__ == "__main__":
    main()
