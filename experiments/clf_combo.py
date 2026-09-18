"""默认段分类概率融合方式与逐目标 clf 权重评估（复用 screen_tree_cache + TCN 工件，纯 numpy）.

检验：p_def 概率平均 vs logit 平均 vs max/min；逐目标 clf_tcn_w 网格。
"""
from __future__ import annotations
import os, sys, warnings, argparse
warnings.filterwarnings("ignore")
import joblib, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.feature import default_mask, load_wells, make_features, make_targets
from src.utils import DEFAULT_VALUES, EPS, TARGETS, TOLERANCE, WEIGHTS
from solution.tcn_model import numpy_forward
RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")

def pointwise(c, pred, true):
    p = np.asarray(pred, float); t = np.asarray(true, float)
    if c == "PERM":
        p = np.log10(np.maximum(p, EPS)); t = np.log10(np.maximum(t, EPS))
    return np.maximum(0.0, 1.0 - np.abs(p - t) / (TOLERANCE[c] * (np.abs(t) + EPS)))

def logit(p): 
    p = np.clip(p, 1e-6, 1-1e-6); return np.log(p/(1-p))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default="T6_long,T8_seed43,T9_seed44")
    ap.add_argument("--wtcn", default="0.60,0.95,0.75")  # POR,PERM,SW
    a = ap.parse_args()
    tags = [t.strip() for t in a.tags.split(",") if t.strip()]
    wmap = dict(zip(TARGETS, [float(x) for x in a.wtcn.split(",")]))
    wells = load_wells("./train", has_labels=True)
    ids = [w["logId"].iloc[0] for w in wells]
    va = [w for w in wells if w["logId"].iloc[0] in set(ids[72:])]
    va_df = pd.concat(va, ignore_index=True)
    Yva = make_targets(va_df); tdef = default_mask(Yva)
    cache = joblib.load(os.path.join(RUNS, "screen_tree_cache.joblib"))
    reg_tree, p_tree = cache["reg"], cache["p_def"]
    rby, pby = {}, {}
    for tag in tags:
        art = joblib.load(os.path.join(RUNS, f"tcn_{tag}.joblib"))
        r, pp = {}, {}
        for w in va:
            Xw = make_features(w, context=False, priors=True).fillna(art["medians"])
            Xw = ((Xw - art["stds"].replace(0, 1.0)) / art["stds"].replace(0, 1.0)).to_numpy(np.float32)
            o = numpy_forward(Xw, art["sd"], art["dilations"])
            for c in TARGETS:
                r.setdefault(c, []).append(o[c][:, 1]); pp.setdefault(c, []).append(1/(1+np.exp(-o[c][:,0])))
        rby[tag] = {c: np.concatenate(v) for c, v in r.items()}
        pby[tag] = {c: np.concatenate(v) for c, v in pp.items()}
    ens = {c: np.mean([rby[t][c] for t in tags], axis=0) for c in TARGETS}
    regs = {c: (1-wmap[c])*reg_tree[c] + wmap[c]*ens[c] for c in TARGETS}
    th_grid = np.arange(0.02, 0.71, 0.02)

    def evalc(name, pmap):
        tot=0.0; parts={}
        for c in TARGETS:
            y = Yva[c].to_numpy(); m = ~np.isnan(y)
            ba, bt = -1, 0.5
            for t in th_grid:
                pred = np.where(pmap[c] >= t, DEFAULT_VALUES[c], np.power(10.0, regs[c]))
                acc = pointwise(c, pred[m], y[m]).mean()
                if acc>ba: ba, bt = acc, float(t)
            pred = np.where(pmap[c] >= bt, DEFAULT_VALUES[c], np.power(10.0, regs[c]))
            d = pointwise(c, pred[m & tdef[c].to_numpy()], y[m & tdef[c].to_numpy()]).mean()
            r = pointwise(c, pred[m & ~tdef[c].to_numpy()], y[m & ~tdef[c].to_numpy()]).mean()
            parts[c] = (ba, r, d, bt); tot += WEIGHTS[c]*ba
        print(f"{name:26s}: {tot*100:.3f} acc=(" + ",".join(f"{parts[c][0]:.4f}" for c in TARGETS)
              + ") real=(" + ",".join(f"{parts[c][1]:.4f}" for c in TARGETS)
              + ") th=(" + ",".join(f"{parts[c][3]:.2f}" for c in TARGETS) + ")")
        return tot*100

    # 逐目标 clf 权重网格（概率平均）
    print("--- 逐目标 clf_tcn_w 网格（prob mean） ---")
    best = {}
    for c in TARGETS:
        rows=[]
        for a_ in np.arange(0.0, 1.01, 0.1):
            pm = (1-a_)*p_tree[c] + a_*np.mean([pby[t][c] for t in tags], axis=0)
            y = Yva[c].to_numpy(); m = ~np.isnan(y)
            ba=-1
            for t in th_grid:
                pred = np.where(pm >= t, DEFAULT_VALUES[c], np.power(10.0, regs[c]))
                ba = max(ba, pointwise(c, pred[m], y[m]).mean())
            rows.append((ba, round(a_,2)))
        rows.sort(reverse=True)
        best[c] = rows[0][1]
        print(f"  {c}: best clf_w={rows[0][1]} acc={rows[0][0]:.4f} | top3 {rows[:3]}")
    pm_best = {c: (1-best[c])*p_tree[c] + best[c]*np.mean([pby[t][c] for t in tags], axis=0) for c in TARGETS}

    print("\n--- 融合方式对比（clf_w=0.5 固定） ---")
    a_ = 0.5
    for name, fn in [
        ("prob mean", lambda c: (1-a_)*p_tree[c] + a_*np.mean([pby[t][c] for t in tags], axis=0)),
        ("logit mean", lambda c: 1/(1+np.exp(-((1-a_)*logit(p_tree[c]) + a_*np.mean([logit(pby[t][c]) for t in tags], axis=0))))),
        ("prob max", lambda c: (1-a_)*p_tree[c] + a_*np.max([pby[t][c] for t in tags], axis=0)),
        ("prob min", lambda c: (1-a_)*p_tree[c] + a_*np.min([pby[t][c] for t in tags], axis=0)),
        ("prob median", lambda c: (1-a_)*p_tree[c] + a_*np.median([pby[t][c] for t in tags], axis=0)),
    ]:
        evalc(name, {c: fn(c) for c in TARGETS})
    print("\n--- 逐目标最优 clf_w（同时用 prob mean） ---")
    evalc("per-target clf_w best", pm_best)

if __name__ == "__main__":
    main()
