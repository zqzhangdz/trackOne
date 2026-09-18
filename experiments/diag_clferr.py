"""分类误差定位：错误点与真实段边界的距离分布（screen 验证井）."""
from __future__ import annotations
import os, sys, warnings
warnings.filterwarnings("ignore")
import joblib, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.feature import default_mask, load_wells, make_features, make_targets
from src.utils import DEFAULT_VALUES, EPS, TARGETS, TOLERANCE, WEIGHTS

def pointwise(c, pred, true):
    p = np.asarray(pred, float); t = np.asarray(true, float)
    if c == "PERM":
        p = np.log10(np.maximum(p, EPS)); t = np.log10(np.maximum(t, EPS))
    return np.maximum(0.0, 1.0 - np.abs(p - t) / (TOLERANCE[c] * (np.abs(t) + EPS)))

def main():
    wells = load_wells("./train", has_labels=True)
    ids = [w["logId"].iloc[0] for w in wells]
    va = [w for w in wells if w["logId"].iloc[0] in set(ids[72:])]
    va_df = pd.concat(va, ignore_index=True)
    Yva = make_targets(va_df)
    tdef = default_mask(Yva)
    cache = joblib.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "screen_tree_cache.joblib"))
    p_tree = cache["p_def"]
    art = joblib.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "tcn_T6_long.joblib"))
    from solution.tcn_model import numpy_forward
    p_tcn = {}
    for w in va:
        Xw = make_features(w, context=False, priors=True).fillna(art["medians"])
        Xw = ((Xw - art["stds"].replace(0, 1.0)) / art["stds"].replace(0, 1.0)).to_numpy(np.float32)
        o = numpy_forward(Xw, art["sd"], art["dilations"])
        for c in TARGETS:
            p_tcn.setdefault(c, []).append(1/(1+np.exp(-o[c][:,0])))
    p_tcn = {c: np.concatenate(v) for c, v in p_tcn.items()}
    larr = va_df["logId"].to_numpy()
    # 逐点距边界距离（按井）
    dist = {c: np.full(len(va_df), 10**6) for c in TARGETS}
    for c in TARGETS:
        td = tdef[c].to_numpy().astype(np.int8)
        for lid in pd.unique(larr):
            m = larr == lid
            b = td[m]
            chg = np.flatnonzero(np.diff(b))
            # 每个点距最近变化点
            for i in range(len(b)):
                if chg.size:
                    dist[c][np.flatnonzero(m)[i]] = int(min(abs(i-j) for j in chg))
    grid = np.arange(0.02, 0.71, 0.02)
    for name, pm in [("tree", p_tree), ("tcn", p_tcn), ("mean", {c: .5*p_tree[c]+.5*p_tcn[c] for c in TARGETS})]:
        parts = []
        for c in TARGETS:
            y = Yva[c].to_numpy(); m = ~np.isnan(y)
            bt, ba = 0.5, -1
            for t in grid:
                pred = np.where(pm[c] >= t, DEFAULT_VALUES[c], np.power(10.0, cache["reg"][c]))
                a = pointwise(c, pred[m], y[m]).mean()
                if a > ba: ba, bt = a, float(t)
            calls = pm[c] >= bt
            err = calls != tdef[c].to_numpy()
            err &= m
            near = err & (dist[c] <= 5)
            far = err & (dist[c] > 5) & (dist[c] < 10**6)
            parts.append(f"{c}: acc={ba:.4f} errN={int(err.sum())} near5={int(near.sum())} far={int(far.sum())}")
        tot = sum(WEIGHTS[c]*0 for c in TARGETS)
        print(f"{name}: " + " | ".join(parts))

if __name__ == "__main__":
    main()
