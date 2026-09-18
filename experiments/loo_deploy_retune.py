"""部署同源工件的融合权重/阈值 LOO 稳健性检验（8 验证井留一）.

对每个留出井：在其余 7 井上按指标调优 (w_tcn, threshold)（clf_w 固定 0.5），
再在留出井上评估；与"当前部署配置(0.60/0.95/0.75, .10/.05/.05)固定应用"对比。
"""
from __future__ import annotations
import os, sys, warnings
warnings.filterwarnings("ignore")
import joblib, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.feature import default_mask, load_wells, make_features, make_targets
from src.utils import DEFAULT_VALUES, EPS, TARGETS, TOLERANCE, WEIGHTS
from solution.tcn_model import numpy_forward
RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
V4_W = {"POR": 0.60, "PERM": 0.95, "SW": 0.75}; V4_TH = {"POR": 0.10, "PERM": 0.05, "SW": 0.05}


def pw(c, pred, true):
    p = np.asarray(pred, float); t = np.asarray(true, float)
    if c == "PERM":
        p = np.log10(np.maximum(p, EPS)); t = np.log10(np.maximum(t, EPS))
    return np.maximum(0.0, 1.0 - np.abs(p - t) / (TOLERANCE[c] * (np.abs(t) + EPS)))


def main():
    wells = load_wells("./train", has_labels=True); ids = [w["logId"].iloc[0] for w in wells]
    va = [w for w in wells if w["logId"].iloc[0] in set(ids[72:])]
    va_df = pd.concat(va, ignore_index=True); Y = make_targets(va_df); td = default_mask(Y)
    larr = va_df["logId"].to_numpy()
    cache = joblib.load(os.path.join(RUNS, "screen_tree_cache.joblib")); rt, pt = cache["reg"], cache["p_def"]
    R, P = {}, {}
    for tag in ["T10_ln10", "T11_seed43_ln10"]:
        a = joblib.load(os.path.join(RUNS, f"tcn_{tag}.joblib")); r, pp = {}, {}
        for w in va:
            X = make_features(w, context=False, priors=True).fillna(a["medians"])
            X = ((X - a["stds"].replace(0, 1.0)) / a["stds"].replace(0, 1.0)).to_numpy(np.float32)
            o = numpy_forward(X, a["sd"], a["dilations"])
            for c in TARGETS:
                r.setdefault(c, []).append(o[c][:, 1]); pp.setdefault(c, []).append(1/(1+np.exp(-o[c][:,0])))
        for c in TARGETS:
            R[c] = R.get(c, 0) + np.concatenate(r[c]) / 2
            P[c] = P.get(c, 0) + np.concatenate(pp[c]) / 2
    grid = np.arange(0.02, 0.71, 0.02); wg = np.arange(0.0, 1.01, 0.05)

    def acc_on(c, mask, w, t):
        y = Y[c].to_numpy(); m = (~np.isnan(y)) & mask
        if m.sum() == 0: return np.nan
        pm = .5*pt[c] + .5*P[c]
        pred = np.where(pm >= t, DEFAULT_VALUES[c], np.power(10., (1-w)*rt[c]+w*R[c]))
        return pw(c, pred[m], y[m]).mean()

    uniq = list(pd.unique(larr))
    rows = []
    for hold in uniq:
        ho = larr == hold; trm = ~ho
        tuned, fix = {}, {}
        for c in TARGETS:
            ba, bw, bt = -1, .7, .5
            for w in wg:
                for t in grid:
                    a = acc_on(c, trm, w, t)
                    if not np.isnan(a) and a > ba: ba, bw, bt = a, float(w), float(t)
            tuned[c] = (bw, bt)
            fix[c] = (V4_W[c], V4_TH[c])
        tot_t = sum(WEIGHTS[c]*acc_on(c, ho, *tuned[c]) for c in TARGETS)*100
        tot_f = sum(WEIGHTS[c]*acc_on(c, ho, *fix[c]) for c in TARGETS)*100
        rows.append((hold, tot_t, tot_f))
        print(f"  well {hold}: tuned={tot_t:.3f} fixed_v4={tot_f:.3f} diff={tot_t-tot_f:+.3f} "
              f"w=({tuned['POR'][0]:.2f},{tuned['PERM'][0]:.2f},{tuned['SW'][0]:.2f})")
    tt = np.array([r[1] for r in rows]); tf = np.array([r[2] for r in rows]); d = tt - tf
    print(f"\nLOO tuned mean={tt.mean():.3f}  fixed_v4 mean={tf.mean():.3f}  gain={d.mean():+.3f}")
    print(f"per-well gain: " + " ".join(f"{x:+.2f}" for x in d) + f"  pos/neg={int((d>0).sum())}/{int((d<0).sum())}")
    print(f"in-sample(all 8 tuned) vs LOO optimism check: LOO gain={d.mean():+.3f}")


if __name__ == "__main__":
    main()
