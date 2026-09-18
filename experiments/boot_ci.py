"""按井 bootstrap 置信区间：判断 screen 增益是否超出噪声（验证井重采样）."""
from __future__ import annotations
import os, sys, warnings
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

def main():
    wells = load_wells("./train", has_labels=True)
    ids = [w["logId"].iloc[0] for w in wells]
    va = [w for w in wells if w["logId"].iloc[0] in set(ids[72:])]
    va_df = pd.concat(va, ignore_index=True)
    Yva = make_targets(va_df)
    cache = joblib.load(os.path.join(RUNS, "screen_tree_cache.joblib"))
    reg_tree, p_tree = cache["reg"], cache["p_def"]
    regs, ps = [], []
    for tag in ("T6_long", "T8_seed43"):
        art = joblib.load(os.path.join(RUNS, f"tcn_{tag}.joblib"))
        r, pp = {}, {}
        for w in va:
            Xw = make_features(w, context=False, priors=True).fillna(art["medians"])
            Xw = ((Xw - art["stds"].replace(0, 1.0)) / art["stds"].replace(0, 1.0)).to_numpy(np.float32)
            o = numpy_forward(Xw, art["sd"], art["dilations"])
            for c in TARGETS:
                r.setdefault(c, []).append(o[c][:, 1]); pp.setdefault(c, []).append(1/(1+np.exp(-o[c][:,0])))
        regs.append({c: np.concatenate(v) for c, v in r.items()})
        ps.append({c: np.concatenate(v) for c, v in pp.items()})
    ens = {c: np.mean([r[c] for r in regs], axis=0) for c in TARGETS}
    pen = {c: np.mean([p[c] for p in ps], axis=0) for c in TARGETS}
    larr = va_df["logId"].to_numpy()
    wid = np.array([np.flatnonzero(larr == lid) for lid in pd.unique(larr)], dtype=object)

    grid = np.arange(0.02, 0.71, 0.02)
    def total_for(regmap, pmap):
        tot = 0.0
        for c in TARGETS:
            y = Yva[c].to_numpy(); m = ~np.isnan(y)
            bt, ba = 0.5, -1
            for t in grid:
                pred = np.where(pmap[c] >= t, DEFAULT_VALUES[c], np.power(10.0, regmap[c]))
                a = pointwise(c, pred[m], y[m]).mean()
                if a > ba: ba, bt = a, float(t)
            tot += WEIGHTS[c]*ba
        return tot*100
    # 需要”逐井可加“近似：用全局调出的阈值，再逐井算 acc
    def per_targets(regmap, pmap):
        ths = {}
        for c in TARGETS:
            y = Yva[c].to_numpy(); m = ~np.isnan(y)
            bt, ba = 0.5, -1
            for t in grid:
                pred = np.where(pmap[c] >= t, DEFAULT_VALUES[c], np.power(10.0, regmap[c]))
                a = pointwise(c, pred[m], y[m]).mean()
                if a > ba: ba, bt = a, float(t)
            ths[c] = bt
        return ths
    def well_scores(regmap, pmap, ths):
        out = []
        for idx in wid:
            s = 0.0; wsum = 0.0
            for c in TARGETS:
                y = Yva[c].to_numpy()[idx]; m = ~np.isnan(y)
                if not m.any():
                    continue
                pred = np.where(pmap[c][idx] >= ths[c], DEFAULT_VALUES[c], np.power(10.0, regmap[c][idx]))
                s += WEIGHTS[c]*pointwise(c, pred[m], y[m]).mean()
                wsum += WEIGHTS[c]
            out.append(s / max(wsum, 1e-9) * 100)
        return np.array(out)

    rng = np.random.default_rng(0)
    WPT = {"POR": 0.65, "PERM": 0.95, "SW": 0.75}
    def mix(rmap):
        return {c: (1.0 - WPT[c]) * reg_tree[c] + WPT[c] * rmap[c] for c in TARGETS}
    cfgs = {
        "v3 (T6 .7 global, tree p)": ({c: 0.3 * reg_tree[c] + 0.7 * regs[0][c] for c in TARGETS}, p_tree),
        "ens2 per-t w, tree p": (mix(ens), p_tree),
        "ens2 per-t w, pmean .5": (mix(ens), {c: 0.5 * p_tree[c] + 0.5 * pen[c] for c in TARGETS}),
    }
    res = {}
    for name, (rm, pm) in cfgs.items():
        ths = per_targets(rm, pm)
        ws = well_scores(rm, pm, ths)
        boot = []
        for _ in range(4000):
            sel = rng.integers(0, len(ws), len(ws))
            boot.append(ws[sel].mean())
        boot = np.array(boot)
        res[name] = boot
        print(f"{name:30s}: point={ws.mean():.3f} boot95=[{np.percentile(boot,2.5):.3f},{np.percentile(boot,97.5):.3f}]")
    base = res["v3 (T6 .7 global, tree p)"]
    for name, boot in res.items():
        if name.startswith("v3"): continue
        d = boot - base
        print(f"Δ({name}) : mean={d.mean():+.3f} boot95=[{np.percentile(d,2.5):+.3f},{np.percentile(d,97.5):+.3f}] P(Δ>0)={np.mean(d>0):.3f}")

if __name__ == "__main__":
    main()
