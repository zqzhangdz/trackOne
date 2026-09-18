"""配对检验：多seed集成相对 v3(T6) 的逐井增益方向与幅度（screen 验证井）."""
from __future__ import annotations
import os, sys, warnings
warnings.filterwarnings("ignore")
import joblib, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.feature import load_wells, make_features, make_targets, default_mask
from src.utils import DEFAULT_VALUES, EPS, TARGETS, TOLERANCE, WEIGHTS
from solution.tcn_model import numpy_forward
RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")

def pw(c, pred, true):
    p = np.asarray(pred, float); t = np.asarray(true, float)
    if c == "PERM":
        p = np.log10(np.maximum(p, EPS)); t = np.log10(np.maximum(t, EPS))
    return np.maximum(0.0, 1.0 - np.abs(p - t) / (TOLERANCE[c] * (np.abs(t) + EPS)))

def main(tags=("T6_long", "T8_seed43")):
    wells = load_wells("./train", has_labels=True)
    ids = [w["logId"].iloc[0] for w in wells]
    va = [w for w in wells if w["logId"].iloc[0] in set(ids[72:])]
    va_df = pd.concat(va, ignore_index=True); Yva = make_targets(va_df)
    cache = joblib.load(os.path.join(RUNS, "screen_tree_cache.joblib"))
    reg_tree, p_tree = cache["reg"], cache["p_def"]
    regs = []
    for tag in tags:
        art = joblib.load(os.path.join(RUNS, f"tcn_{tag}.joblib")); r = {}
        for w in va:
            Xw = make_features(w, context=False, priors=True).fillna(art["medians"])
            Xw = ((Xw - art["stds"].replace(0, 1.0)) / art["stds"].replace(0, 1.0)).to_numpy(np.float32)
            o = numpy_forward(Xw, art["sd"], art["dilations"])
            for c in TARGETS: r.setdefault(c, []).append(o[c][:, 1])
        regs.append({c: np.concatenate(v) for c, v in r.items()})
    ens = {c: np.mean([r[c] for r in regs], axis=0) for c in TARGETS}
    WPT = {"POR": 0.65, "PERM": 0.95, "SW": 0.75}
    mix = {c: (1-WPT[c])*reg_tree[c] + WPT[c]*ens[c] for c in TARGETS}
    v3 = {c: 0.3*reg_tree[c] + 0.7*regs[0][c] for c in TARGETS}
    larr = va_df["logId"].to_numpy()
    grid = np.arange(0.02, 0.71, 0.02)
    def ths_for(rm, pm):
        out = {}
        for c in TARGETS:
            y = Yva[c].to_numpy(); m = ~np.isnan(y); bt, ba = 0.5, -1
            for t in grid:
                pred = np.where(pm[c] >= t, DEFAULT_VALUES[c], np.power(10.0, rm[c]))
                a = pw(c, pred[m], y[m]).mean()
                if a > ba: ba, bt = a, float(t)
            out[c] = bt
        return out
    def per_well(rm, pm, ths):
        res = []
        for lid in pd.unique(larr):
            idx = larr == lid; s = 0.0; wsum = 0.0
            for c in TARGETS:
                y = Yva[c].to_numpy()[idx]; m = ~np.isnan(y)
                if not m.any(): continue
                pred = np.where(pm[c][idx] >= ths[c], DEFAULT_VALUES[c], np.power(10.0, rm[c][idx]))
                s += WEIGHTS[c]*pw(c, pred[m], y[m]).mean(); wsum += WEIGHTS[c]
            res.append(s/max(wsum,1e-9)*100)
        return np.array(res)
    t_v3 = ths_for(v3, p_tree); t_mix = ths_for(mix, {c: .5*p_tree[c]+.5*np.mean([0],0) if False else p_tree[c] for c in TARGETS})
    # 用 tree p_def（保守）
    a = per_well(v3, p_tree, t_v3)
    b = per_well(mix, p_tree, t_mix)
    print("well, v3, ens3, delta")
    for lid, x, y in zip(pd.unique(larr), a, b):
        print(f"  {lid[:8]}: {x:.3f} -> {y:.3f}  Δ={y-x:+.3f}")
    d = b - a
    print(f"mean Δ={d.mean():+.3f}  n_pos={int((d>0).sum())}/{len(d)}  min={d.min():+.3f} max={d.max():+.3f}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--tags", default="T6_long,T8_seed43")
    t = [x.strip() for x in ap.parse_args().tags.split(",") if x.strip()]
    main(tuple(t))
