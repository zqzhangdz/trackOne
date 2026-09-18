"""逐目标融合权重 + 决策诊断（screen 验证井，复用 screen_tree_cache）.

分解口径与 EXPERIMENTS.md 一致：acc_def 在“真实默认标签”点上统计。
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
    cache = joblib.load(os.path.join(RUNS, "screen_tree_cache.joblib"))
    reg_tree, p_tree = cache["reg"], cache["p_def"]

    reg = {}
    p_tcn = {}
    for tag in ("T6_long", "T8_seed43", "T9_seed44"):
        fp = os.path.join(RUNS, f"tcn_{tag}.joblib")
        if not os.path.exists(fp): continue
        art = joblib.load(fp)
        r, pp = {}, {}
        for w in va:
            Xw = make_features(w, context=False, priors=True).fillna(art["medians"])
            Xw = ((Xw - art["stds"].replace(0, 1.0)) / art["stds"].replace(0, 1.0)).to_numpy(np.float32)
            o = numpy_forward(Xw, art["sd"], art["dilations"])
            for c in TARGETS:
                r.setdefault(c, []).append(o[c][:, 1]); pp.setdefault(c, []).append(1/(1+np.exp(-o[c][:,0])))
        reg[tag] = {c: np.concatenate(v) for c, v in r.items()}
        p_tcn[tag] = {c: np.concatenate(v) for c, v in pp.items()}
        print(f"loaded {tag}")

    grid = np.arange(0.02, 0.71, 0.02)
    def best_acc(c, regmap, pmap):
        y = Yva[c].to_numpy(); m = ~np.isnan(y)
        bt, ba = 0.5, -1
        for t in grid:
            pred = np.where(pmap[c] >= t, DEFAULT_VALUES[c], np.power(10.0, regmap[c]))
            a = pointwise(c, pred[m], y[m]).mean()
            if a > ba: ba, bt = a, float(t)
        return ba, bt
    def dec_of(c, regmap, pmap, t):
        y = Yva[c].to_numpy(); m = ~np.isnan(y)
        pred = np.where(pmap[c] >= t, DEFAULT_VALUES[c], np.power(10.0, regmap[c]))
        return {"acc": pointwise(c, pred[m], y[m]).mean(),
                "def": pointwise(c, pred[m & tdef[c].to_numpy()], y[m & tdef[c].to_numpy()]).mean(),
                "real": pointwise(c, pred[m & ~tdef[c].to_numpy()], y[m & ~tdef[c].to_numpy()]).mean()}

    ens = {c: np.mean([reg[t][c] for t in reg], axis=0) for c in TARGETS}

    print("\n--- 逐目标 weight (tree vs ens) 扫描 ---")
    best_w = {}
    for c in TARGETS:
        rows = []
        for w in np.arange(0.0, 1.01, 0.05):
            rg = (1-w)*reg_tree[c] + w*ens[c]
            a, t = best_acc(c, {c: rg}, p_tree)
            rows.append((a, w, t))
        rows.sort(reverse=True)
        print(f"{c}: best w_tcn={rows[0][1]:.2f} acc={rows[0][0]:.4f} th={rows[0][2]:.2f} | top3 " +
              ", ".join(f"w{w:.2f}:{a:.4f}" for a, w, _ in rows[:3]))
        best_w[c] = rows[0][1]

    def report(name, regmap, pmap):
        tot = 0.0; parts = {}
        for c in TARGETS:
            a, t = best_acc(c, regmap, pmap)
            d = dec_of(c, regmap, pmap, t)
            parts[c] = (d, t); tot += WEIGHTS[c]*d["acc"]
        print(f"{name:24s}: {tot*100:.3f} acc=(" + ",".join(f"{parts[c][0]['acc']:.4f}" for c in TARGETS)
              + ") real=(" + ",".join(f"{parts[c][0]['real']:.4f}" for c in TARGETS)
              + ") def=(" + ",".join(f"{parts[c][0]['def']:.4f}" for c in TARGETS)
              + ") th=(" + ",".join(f"{parts[c][1]:.2f}" for c in TARGETS) + ")")
        return tot*100

    report("tree only", reg_tree, p_tree)
    report("T6 0.7 global", {c: 0.3*reg_tree[c] + 0.7*reg["T6_long"][c] for c in TARGETS}, p_tree )
    if "T8_seed43" in reg and "T9_seed44" in reg:
        report("ens 0.7 global", {c: 0.3*reg_tree[c] + 0.7*ens[c] for c in TARGETS}, p_tree)
        report("ens per-target w", {c: (1-best_w[c])*reg_tree[c] + best_w[c]*ens[c] for c in TARGETS}, p_tree)
    # TCN p_def 融合决策
    for a in (0.3, 0.5, 0.7):
        pm = {c: (1-a)*p_tree[c] + a*np.mean([p_tcn[t][c] for t in p_tcn], axis=0) for c in TARGETS}
        report(f"ens per-t + pmean{a:.1f}", {c: (1-best_w[c])*reg_tree[c] + best_w[c]*ens[c] for c in TARGETS}, pm)

if __name__ == "__main__":
    main()
