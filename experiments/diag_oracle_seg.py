"""分类headroom与回归上界诊断（screen 验证井，复用 screen_tree_cache）.

regmap 一律为 log10 空间(POR/SW 也取 log10 以便统一合成)。
输出：
  A: tree 决策 + tree 回归（当前树系基线）
  B: ORACLE 分段 + tree 回归（分类 headroom）
  C: ORACLE 分段 + ORACLE 逐连续段均值（level 上界）
  D: ORACLE 分段 + ORACLE 真值（总体上界）
"""
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
    cache = joblib.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "screen_tree_cache.joblib"))
    assert list(cache["va_ids"]) == ids[72:]
    reg_tree = cache["reg"]; p_def_tree = cache["p_def"]

    tcn_reg = None
    tp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "tcn_T6_long.joblib")
    if os.path.exists(tp):
        from solution.tcn_model import numpy_forward
        art = joblib.load(tp)
        tcn_reg = {}
        for w in va:
            Xw = make_features(w, context=False, priors=True).fillna(art["medians"])
            Xw = ((Xw - art["stds"].replace(0, 1.0)) / art["stds"].replace(0, 1.0)).to_numpy(np.float32)
            o = numpy_forward(Xw, art["sd"], art["dilations"])
            for c in TARGETS:
                tcn_reg.setdefault(c, []).append(o[c][:, 1])
        tcn_reg = {c: np.concatenate(v) for c, v in tcn_reg.items()}

    larr = va_df["logId"].to_numpy()
    true_def = {c: default_mask(Yva)[c].to_numpy() for c in TARGETS}
    ylog = {}
    for c in TARGETS:
        y = Yva[c].to_numpy()
        ylog[c] = np.log10(np.maximum(y, EPS)) if c in ("PERM",) else y  # 统一到 log10
        if c != "PERM":
            ylog[c] = np.log10(np.maximum(y, EPS))

    def rep(name, dm, regmap, thresholds=None):
        dec = {}
        for c in TARGETS:
            y = Yva[c].to_numpy(); mm = ~np.isnan(y)
            if thresholds is not None:
                d = p_def_tree[c] >= thresholds[c] if dm is None else dm[c]
            else:
                d = np.asarray(dm[c], bool)
            pred = np.where(d, np.log10(DEFAULT_VALUES[c]), regmap[c])
            pred_phys = np.power(10.0, pred)
            dec[c] = {
                "acc": pointwise(c, pred_phys[mm], y[mm]).mean(),
                "acc_def": pointwise(c, pred_phys[mm & d], y[mm & d]).mean(),
                "acc_real": pointwise(c, pred_phys[mm & ~d], y[mm & ~d]).mean(),
            }
        tot = sum(WEIGHTS[c]*dec[c]["acc"] for c in TARGETS)*100
        print(f"{name:36s}: {tot:.3f} real=(" + ",".join(f"{dec[c]['acc_real']:.4f}" for c in TARGETS)
              + ") def=(" + ",".join(f"{dec[c]['acc_def']:.4f}" for c in TARGETS) + ")")
        return tot

    rep("A tree dec + tree reg", None, reg_tree, thresholds={"POR":.10,"PERM":.05,"SW":.05})
    rep("B ORACLE seg + tree reg", true_def, reg_tree)
    if tcn_reg is not None:
        rep("B2 ORACLE seg + TCN reg", true_def, tcn_reg)
    # C: oracle 逐连续段均值（level 上界）
    segmean = {}
    for c in TARGETS:
        arr = np.full(len(va_df), np.nan)
        ya = ylog[c]
        for lid in pd.unique(larr):
            msk = larr == lid
            d = msk & true_def[c]
            r = msk & ~true_def[c] & ~np.isnan(ya)
            arr[d] = np.log10(DEFAULT_VALUES[c])
            # 连续真实段
            ri = np.flatnonzero(r)
            if ri.size:
                brk = np.flatnonzero(np.diff(ri) != 1)
                groups = np.split(ri, brk + 1)
                for g in groups:
                    v = ya[g]
                    arr[g] = np.nanmean(v)
            arr[~np.isnan(arr)] = arr[~np.isnan(arr)]
        segmean[c] = arr
    rep("C ORACLE seg + per-seg mean", true_def, segmean)
    rep("D ORACLE seg + ORACLE truth", true_def, ylog)
    if tcn_reg is not None:
        rep("E ORACLE seg + mean(tree,TCN)", true_def, {c: 0.5*reg_tree[c]+0.5*tcn_reg[c] for c in TARGETS})

if __name__ == "__main__":
    main()
