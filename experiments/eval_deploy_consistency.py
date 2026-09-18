"""部署一致性诊断：当前 model.pkl 的 TCN 由 train.py(正确 LN10) 训练，但融合权重/阈值
是在 run_tcn.py(buggy LN10) 工件上调出的 -> 存在训练/调优不同源。本脚本在
"部署同源"工件(T10=seed42, T11=seed43, 正确 LN10) 上重新调优权重/阈值，
并用逐井配对 bootstrap 判断相对当前部署配置(v4 固定 w/th)的增益是否可信。
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
V4_W = {"POR": 0.60, "PERM": 0.95, "SW": 0.75}
V4_TH = {"POR": 0.10, "PERM": 0.05, "SW": 0.05}


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

    def load(tags):
        R, P = {}, {}
        for tag in tags:
            a = joblib.load(os.path.join(RUNS, f"tcn_{tag}.joblib")); r, pp = {}, {}
            for w in va:
                X = make_features(w, context=False, priors=True).fillna(a["medians"])
                X = ((X - a["stds"].replace(0, 1.0)) / a["stds"].replace(0, 1.0)).to_numpy(np.float32)
                o = numpy_forward(X, a["sd"], a["dilations"])
                for c in TARGETS:
                    r.setdefault(c, []).append(o[c][:, 1]); pp.setdefault(c, []).append(1/(1+np.exp(-o[c][:,0])))
            for c in TARGETS:
                R[c] = R.get(c, 0) + np.concatenate(r[c]) / len(tags)
                P[c] = P.get(c, 0) + np.concatenate(pp[c]) / len(tags)
        return R, P

    grid = np.arange(0.02, 0.71, 0.02); wg = np.arange(0.0, 1.01, 0.05)

    def tune(R, P, clf_w=0.5):
        ws, ths = {}, {}
        for c in TARGETS:
            y = Y[c].to_numpy(); m = ~np.isnan(y)
            pm = (1-clf_w)*pt[c] + clf_w*P[c]
            ba, bt, bw = -1, .5, .7
            for w in wg:
                for t in grid:
                    pred = np.where(pm >= t, DEFAULT_VALUES[c], np.power(10., (1-w)*rt[c]+w*R[c]))
                    a = pw(c, pred[m], y[m]).mean()
                    if a > ba: ba, bt, bw = a, float(t), float(w)
            ws[c], ths[c] = bw, bt
        return ws, ths

    def per_well(regs, pdef, ws, ths, clf_w=0.5):
        out = {}
        for lid in pd.unique(larr):
            mw = larr == lid; tot = 0.0
            for c in TARGETS:
                y = Y[c].to_numpy(); m = (~np.isnan(y)) & mw
                if m.sum() == 0: continue
                pm = (1-clf_w)*pt[c] + clf_w*pdef[c]
                pred = np.where(pm >= ths[c], DEFAULT_VALUES[c], np.power(10., (1-ws[c])*rt[c]+ws[c]*regs[c]))
                tot += WEIGHTS[c]*pw(c, pred[m], y[m]).mean()
            out[lid] = tot*100
        return out

    def total(regs, pdef, ws, ths, clf_w=0.5):
        return float(np.mean(list(per_well(regs, pdef, ws, ths, clf_w).values())))

    for name, tags in [("deploy-src T10+T11 (corrected)", ["T10_ln10", "T11_seed43_ln10"]),
                       ("screen-src T6+T8 (buggy)", ["T6_long", "T8_seed43"])]:
        R, P = load(tags)
        ws, ths = tune(R, P)
        t_tuned = total(R, P, ws, ths)
        t_v4 = total(R, P, V4_W, V4_TH)
        print(f"\n{name}")
        print(f"  tuned w=({ws['POR']:.2f},{ws['PERM']:.2f},{ws['SW']:.2f}) th=({ths['POR']:.2f},{ths['PERM']:.2f},{ths['SW']:.2f}) -> total={t_tuned:.3f}")
        print(f"  current v4 w (0.60,0.95,0.75) th (0.10,0.05,0.05)          -> total={t_v4:.3f}")
        pb = per_well(R, P, V4_W, V4_TH); pt_ = per_well(R, P, ws, ths)
        diff = np.array([pt_[k]-pb[k] for k in sorted(pb)])
        rng = np.random.default_rng(0)
        boot = np.array([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(5000)])
        print(f"  retune gain={diff.mean():+.3f} boot95%=[{np.percentile(boot,2.5):+.3f},{np.percentile(boot,97.5):+.3f}] pos/neg={int((diff>0).sum())}/{int((diff<0).sum())}")


if __name__ == "__main__":
    main()
