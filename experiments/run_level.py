"""M5 层级模型 screen 评估：段级 level(段聚合特征) + 点级 wiggle(局部测井偏差).

变体:
  L0 tree blend (现有最终) 参考
  L1 level-only (真段)
  L2 level + 线性 wiggle (真段)
  L3 level + 线性 wiggle (预测段)
  L4 0.5*(L3) + 0.5*(tree+T3+T5 blend)
"""
from __future__ import annotations
import sys, os, warnings
warnings.filterwarnings("ignore")
import joblib, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.feature import (default_mask, inverse_transform_target, load_wells, make_features,
                         make_targets, transform_target)
from src.model import TwoStageModel
from src.utils import DEFAULT_VALUES, EPS, TARGETS, TOLERANCE, WEIGHTS
from solution.tcn_model import numpy_forward


def pointwise(c, pred, true):
    p = np.asarray(pred, float)
    t = np.asarray(true, float)
    if c == "PERM":
        p = np.log10(np.maximum(p, EPS))
        t = np.log10(np.maximum(t, EPS))
    return np.maximum(0.0, 1.0 - np.abs(p - t) / (TOLERANCE[c] * (np.abs(t) + EPS)))


def decompose(c, pred, y, isdef):
    valid = ~np.isnan(y)
    d = valid & isdef
    r = valid & ~isdef
    return {
        "acc": pointwise(c, pred[valid], y[valid]).mean(),
        "acc_def": pointwise(c, pred[d], y[d]).mean() if d.sum() else float("nan"),
        "acc_real": pointwise(c, pred[r], y[r]).mean() if r.sum() else float("nan"),
    }

RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
LOGCOLS = ["GR", "PE", "SP", "CAL", "AC", "DEN", "CNL", "logRT", "logRXO", "POR_D", "POR_S", "POR_avg"]


def segments_of(mask: np.ndarray):
    idx = np.flatnonzero(np.diff(np.concatenate([[0], mask.view(np.int8), [0]])))
    return list(zip(idx[::2], idx[1::2]))


def seg_features(X: pd.DataFrame, s: int, e: int) -> dict:
    g = X.iloc[s:e][LOGCOLS]
    f = {}
    for col in LOGCOLS:
        f[f"m_{col}"] = float(g[col].mean(skipna=True))
        f[f"s_{col}"] = float(g[col].std(skipna=True) or 0.0)
    f["len"] = float(e - s)
    f["log_len"] = float(np.log(e - s))
    return f


def main(use_tcn_blend: bool = True):
    print("[1] load", flush=True)
    wells = load_wells("./train", has_labels=True)
    ids = [w["logId"].iloc[0] for w in wells]
    tr = [w for w in wells if w["logId"].iloc[0] in set(ids[:72])]
    va = [w for w in wells if w["logId"].iloc[0] in set(ids[72:])]
    tr_df, va_df = pd.concat(tr, ignore_index=True), pd.concat(va, ignore_index=True)
    Xtr = make_features(tr_df, context=True, priors=True)
    Xva = make_features(va_df, context=True, priors=True)
    Ytr, Yva = make_targets(tr_df), make_targets(va_df)
    dtr = default_mask(Ytr)
    print("[2] features", flush=True)
    tree = TwoStageModel(monotonic=True).fit(Xtr, Ytr, dtr)
    print("[3] tree fit", flush=True)
    p_def_va = {c: tree.classifiers[c].predict_proba(Xva)[:, 1] for c in TARGETS}
    p_def_tr = {c: tree.classifiers[c].predict_proba(Xtr)[:, 1] for c in TARGETS}
    print("[4] pdef", flush=True)
    reg_tree_log = {c: np.log10(np.maximum(inverse_transform_target(
        c, pd.Series(tree.regressors[c].predict(Xva), index=Xva.index)).to_numpy(), EPS)) for c in TARGETS}

    print("[5] tree reg", flush=True)
    tcn_regs = {}
    if use_tcn_blend:
        va_ids = va_df["logId"].to_numpy()
        art3 = joblib.load(os.path.join(RUNS, "tcn_T3_sel80.joblib"))
        art5 = joblib.load(os.path.join(RUNS, "tcn_T5_pass2.joblib"))
        ch = joblib.load(os.path.join(RUNS, f"pass1_val_{art5['pass1_tag']}.joblib"))
        ch = {k: np.stack([v[c] for c in TARGETS], axis=1) for k, v in ch.items()}
        for tag, art, channels in (("t3", art3, None), ("t5", art5, ch)):
            medians, stds = art["medians"], art["stds"]
            regs = {c: np.zeros(len(va_df)) for c in TARGETS}
            for w in va:
                Xw = make_features(w, context=False, priors=True).fillna(medians)
                Xw = ((Xw - stds.replace(0, 1.0)) / stds.replace(0, 1.0)).to_numpy(dtype=np.float32)
                if channels is not None:
                    Xw = np.concatenate([Xw, channels[w["logId"].iloc[0]]], axis=1).astype(np.float32)
                o = numpy_forward(Xw, art["sd"], art["dilations"])
                m = va_ids == w["logId"].iloc[0]
                for c in TARGETS:
                    regs[c][m] = o[c][:, 1]
            tcn_regs[tag] = regs
    blend_log = {c: 0.4 * reg_tree_log[c] + 0.3 * tcn_regs.get("t3", {c: reg_tree_log[c]})[c]
                 + 0.3 * tcn_regs.get("t5", {c: reg_tree_log[c]})[c] for c in TARGETS}

    # ---- 训练 level 模型与 wiggle 系数（训练井, 真段）----
    print("[6] tcn regs", flush=True)
    from sklearn.ensemble import HistGradientBoostingRegressor

    level_models, wiggle_coefs = {}, {}
    Xtr_base = make_features(tr_df, context=False, priors=True)
    tr_ids_arr = tr_df["logId"].to_numpy()
    for c in TARGETS:
        rows, targs = [], []
        wig_x, wig_y = [], []
        for w in tr:
            lid = w["logId"].iloc[0]
            Yw = make_targets(w); isdw = default_mask(Yw)
            Xw = make_features(w, context=False, priors=True)
            y = Yw[c].to_numpy(); valid = ~np.isnan(y)
            real = valid & ~isdw[c].to_numpy()
            t = transform_target(c, pd.Series(np.where(valid, y, np.nan))).to_numpy()
            for (s, e) in segments_of(real):
                if e - s < 3:
                    continue
                rows.append(seg_features(Xw, s, e))
                targs.append(float(np.nanmean(t[s:e])))
                mx = Xw.iloc[s:e][LOGCOLS].mean(numeric_only=True).to_numpy()
                dx = np.nan_to_num(Xw.iloc[s:e][LOGCOLS].to_numpy() - mx)
                wig_x.append(dx)
                wig_y.append(t[s:e] - np.nanmean(t[s:e]))
        dfseg = pd.DataFrame(rows)
        lm = HistGradientBoostingRegressor(loss="squared_error", learning_rate=0.08, max_iter=300,
                                           max_leaf_nodes=15, min_samples_leaf=20, random_state=42)
        lm.fit(dfseg.to_numpy(dtype=np.float64), np.array(targs))
        level_models[c] = (lm, list(dfseg.columns))
        Xw_ = np.vstack(wig_x); yw_ = np.concatenate(wig_y)
        A = np.hstack([Xw_, np.ones((len(Xw_), 1))])
        coef, *_ = np.linalg.lstsq(A, yw_, rcond=None)
        wiggle_coefs[c] = coef

    # ---- 验证井上构造 level/wiggle 预测 ----
    def predict_level_wiggle(w, segs):
        Xw = make_features(w, context=False, priors=True)
        n = len(Xw)
        lvl = {c: np.full(n, np.nan) for c in TARGETS}
        wig = {c: np.zeros(n) for c in TARGETS}
        for c in TARGETS:
            lm, cols = level_models[c]
            for (s, e) in segs:
                f = pd.DataFrame([seg_features(Xw, s, e)])[cols].to_numpy(dtype=np.float64)
                lvl[c][s:e] = lm.predict(f)[0]
                mx = Xw.iloc[s:e][LOGCOLS].mean(numeric_only=True).to_numpy()
                dx = np.nan_to_num(Xw.iloc[s:e][LOGCOLS].to_numpy() - mx)
                wig[c][s:e] = np.hstack([dx, np.ones((e - s, 1))]) @ wiggle_coefs[c]
        return lvl, wig

    va_ids = va_df["logId"].to_numpy()
    grid = np.arange(0.05, 0.96, 0.05)

    def score_regs(regs, name):
        th, dec = {}, {}
        for c in TARGETS:
            y = Yva[c].to_numpy(); m = ~np.isnan(y)
            best_t, best_a = 0.5, -1.0
            for t in grid:
                pred = np.where(p_def_va[c] >= t, DEFAULT_VALUES[c], np.power(10.0, regs[c]))
                a = pointwise(c, pred[m], y[m]).mean()
                if a > best_a:
                    best_a, best_t = a, float(t)
            th[c] = best_t
            pred = np.where(p_def_va[c] >= best_t, DEFAULT_VALUES[c], np.power(10.0, regs[c]))
            isdef = default_mask(Yva)[c].to_numpy() & m
            dec[c] = decompose(c, pred, y, isdef)
        tot = sum(WEIGHTS[c] * dec[c]["acc"] for c in TARGETS) * 100
        print(f"{name:26s}: total={tot:.3f} | " + " ".join(f"{c} real={dec[c]['acc_real']:.4f}" for c in TARGETS))
        return tot

    score_regs(blend_log, "L0 current blend")

    variants = {}
    for segmode in ("true", "pred"):
        lvl_all = {c: np.zeros(len(va_df)) for c in TARGETS}
        wig_all = {c: np.zeros(len(va_df)) for c in TARGETS}
        for w in va:
            lid = w["logId"].iloc[0]
            m = va_ids == lid
            nw = m.sum()
            if segmode == "true":
                Yw = make_targets(w); isdw = default_mask(Yw)
                segs_per = {}
                for c in TARGETS:
                    real = (~np.isnan(Yw[c].to_numpy())) & ~isdw[c].to_numpy()
                    segs_per[c] = segments_of(real)
            else:
                segs_per = {}
                for c in TARGETS:
                    pm = p_def_va[c][m] < 0.10
                    pm = pd.Series(pm.astype(np.int8)).rolling(9, center=True, min_periods=1).mean().to_numpy() >= 0.5
                    segs_per[c] = segments_of(pm)
            for c in TARGETS:
                lvl, wig = predict_level_wiggle(w, segs_per[c])
                lvl_all[c][m] = lvl[c]
                wig_all[c][m] = wig[c]
        lw_log, fallback = {}, {}
        for c in TARGETS:
            phys = inverse_transform_target(c, pd.Series(np.nan_to_num(lvl_all[c] + wig_all[c], nan=0.0))).to_numpy()
            lw_log[c] = np.log10(np.maximum(phys, EPS))
            fallback[c] = blend_log[c]
        lvl_phys = {c: inverse_transform_target(c, pd.Series(np.nan_to_num(lvl_all[c], nan=0.0))).to_numpy() for c in TARGETS}
        lvl_log = {c: np.log10(np.maximum(lvl_phys[c], EPS)) for c in TARGETS}
        for c in TARGETS:
            nanm = ~np.isfinite(lvl_log[c])
            lvl_log[c][nanm] = fallback[c][nanm]
            lw_log[c][nanm] = fallback[c][nanm]
        variants[f"L1 level ({segmode})"] = lvl_log
        variants[f"L2 level+wig ({segmode})"] = lw_log
        variants[f"L4 0.5L3+0.5blend ({segmode})"] = {
            c: 0.5 * lw_log[c] + 0.5 * blend_log[c] for c in TARGETS}
    for name, regs in variants.items():
        score_regs(regs, name)


if __name__ == "__main__":
    main()
