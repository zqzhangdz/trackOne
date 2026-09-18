"""v4 融合协议下的组合×clf权重扫描（TCN 前向缓存到 npz，重复评估秒级）。

协议（与 eval_v4/eval_pertarget/eval_seedcombos 一致，纯 numpy）：
  1) 逐目标在 (w_tcn, clf_w, threshold) 网格上最大化 acc（在验证井内调优）；
  2) 报告 total 与 acc_def/acc_real 分解（acc_def 在真实默认标签点上统计）。
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
CACHE = os.path.join(RUNS, "tcn_screen_outputs.npz")


def pw(c, pred, true):
    p = np.asarray(pred, float); t = np.asarray(true, float)
    if c == "PERM":
        p = np.log10(np.maximum(p, EPS)); t = np.log10(np.maximum(t, EPS))
    return np.maximum(0.0, 1.0 - np.abs(p - t) / (TOLERANCE[c] * (np.abs(t) + EPS)))


def collect(va, tags):
    store = {}
    if os.path.exists(CACHE):
        store = {k: v for k, v in np.load(CACHE).items()}
    missing = [t for t in tags if f"{t}__POR__reg" not in store]
    if missing:
        for tag in missing:
            fp = os.path.join(RUNS, f"tcn_{tag}.joblib")
            if not os.path.exists(fp):
                continue
            art = joblib.load(fp)
            r, pp = {}, {}
            for w in va:
                Xw = make_features(w, context=False, priors=True).fillna(art["medians"])
                Xw = ((Xw - art["stds"].replace(0, 1.0)) / art["stds"].replace(0, 1.0)).to_numpy(np.float32)
                o = numpy_forward(Xw, art["sd"], art["dilations"])
                for c in TARGETS:
                    r.setdefault(c, []).append(o[c][:, 1])
                    pp.setdefault(c, []).append(1/(1+np.exp(-o[c][:, 0])))
            for c in TARGETS:
                store[f"{tag}__{c}__reg"] = np.concatenate(r[c])
                store[f"{tag}__{c}__pdf"] = np.concatenate(pp[c])
            print(f"computed {tag}")
        np.savez_compressed(CACHE, **store)
    return store


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--combos", required=True)
    ap.add_argument("--tags", default="T6_long,T8_seed43,T9_seed44,T10_ln10,T11_seed43_ln10")
    ap.add_argument("--clf_ws", default="0.3,0.5,0.7")
    a = ap.parse_args()
    tags = [t.strip() for t in a.tags.split(",") if t.strip()]
    clf_ws = [float(x) for x in a.clf_ws.split(",") if x.strip()]
    wells = load_wells("./train", has_labels=True); ids = [w["logId"].iloc[0] for w in wells]
    va = [w for w in wells if w["logId"].iloc[0] in set(ids[72:])]
    va_df = pd.concat(va, ignore_index=True); Yva = make_targets(va_df); tdef = default_mask(Yva)
    store = collect(va, tags)
    cache = joblib.load(os.path.join(RUNS, "screen_tree_cache.joblib")); reg_tree, p_tree = cache["reg"], cache["p_def"]
    th_grid = np.arange(.02, .71, .02); w_grid = np.arange(0.0, 1.01, .05)

    def ev(tags_sel, clf_w):
        tot = 0.0; parts = {}; cfg = {}
        for c in TARGETS:
            ens = np.mean([store[f"{t}__{c}__reg"] for t in tags_sel], axis=0)
            pens = np.mean([store[f"{t}__{c}__pdf"] for t in tags_sel], axis=0)
            y = Yva[c].to_numpy(); m = ~np.isnan(y)
            pm = (1 - clf_w) * p_tree[c] + clf_w * pens
            ba, bt, bw = -1, .5, .7
            for w in w_grid:
                reg = (1 - w) * reg_tree[c] + w * ens
                for t in th_grid:
                    pred = np.where(pm >= t, DEFAULT_VALUES[c], np.power(10., reg))
                    acc = pw(c, pred[m], y[m]).mean()
                    if acc > ba: ba, bt, bw = acc, float(t), float(w)
            reg = (1 - bw) * reg_tree[c] + bw * ens
            pred = np.where(pm >= bt, DEFAULT_VALUES[c], np.power(10., reg))
            d = pw(c, pred[m & tdef[c].to_numpy()], y[m & tdef[c].to_numpy()]).mean()
            r = pw(c, pred[m & ~tdef[c].to_numpy()], y[m & ~tdef[c].to_numpy()]).mean()
            parts[c] = (ba, r, d); cfg[c] = (bw, bt); tot += WEIGHTS[c] * ba
        return tot * 100, parts, cfg

    print(f"{'combo':22s} {'clf_w':>5s} {'total':>7s}  real(POR,PERM,SW)        def(POR,PERM,SW)        w(POR,PERM,SW)")
    for combo in a.combos.split("|"):
        sel = [t.strip() for t in combo.split(",") if t.strip() and f"{t.strip()}__POR__reg" in store]
        if not sel: continue
        label = "+".join(t.replace("_long", "").replace("_seed", "").replace("_ln10", "").replace("_43", "") for t in sel)
        for cw in clf_ws:
            t, parts, cfg = ev(sel, cw)
            print(f"{label:22s} {cw:5.2f} {t:7.3f}  ({parts['POR'][1]:.4f},{parts['PERM'][1]:.4f},{parts['SW'][1]:.4f}) "
                  f"({parts['POR'][2]:.4f},{parts['PERM'][2]:.4f},{parts['SW'][2]:.4f}) "
                  f"({cfg['POR'][0]:.2f},{cfg['PERM'][0]:.2f},{cfg['SW'][0]:.2f})")


if __name__ == "__main__":
    main()
