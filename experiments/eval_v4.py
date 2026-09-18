"""v4 评估：多 seed TCN 集成 + 逐目标融合权重 + p_def 决策组合（全部在 screen 验证井内调优）.

不导入 torch（避免与 sklearn OpenMP 冲突）：TCN 前向用纯 numpy。
"""
from __future__ import annotations

import argparse
import itertools
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.feature import (default_mask, inverse_transform_target, load_wells, make_features,
                         make_targets, transform_target)
from src.model import TwoStageModel
from src.utils import DEFAULT_VALUES, EPS, TARGETS, TOLERANCE, WEIGHTS
from solution.tcn_model import numpy_forward

RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")


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


def build_tree(tr_df, weight_tol=True):
    Xtr = make_features(tr_df, context=True, priors=True)
    Ytr = make_targets(tr_df)
    model = TwoStageModel(monotonic=True).fit(Xtr, Ytr, default_mask(Ytr))
    if weight_tol:
        for c in TARGETS:
            valid = Ytr[c].notna().to_numpy()
            y = Ytr.loc[valid, c]
            isdef = default_mask(Ytr).loc[valid, c].astype(int).to_numpy()
            real = isdef == 0
            yv = y.to_numpy()[real]
            w = np.abs(np.log10(np.maximum(yv, EPS))) if c == "PERM" else np.abs(yv)
            w = np.clip(w, np.percentile(w, 5), np.percentile(w, 95))
            w = w / w.mean()
            model.regressors[c].fit(Xtr.loc[valid][real], transform_target(c, y[real]), sample_weight=w)
    return model, Xtr


def tcn_outputs(wells, art):
    """逐井 numpy 前向，返回 {c: (reg_log10, p_def)} 按 va_df 行顺序拼接."""
    medians, stds = art["medians"], art["stds"]
    reg, pdf = {}, {}
    for w in wells:
        Xw = make_features(w, context=False, priors=True).fillna(medians)
        Xw = ((Xw - stds.replace(0, 1.0)) / stds.replace(0, 1.0)).to_numpy(dtype=np.float32)
        o = numpy_forward(Xw, art["sd"], art["dilations"])
        for c in TARGETS:
            reg.setdefault(c, []).append(o[c][:, 1])
            pdf.setdefault(c, []).append(1.0 / (1.0 + np.exp(-o[c][:, 0])))
    return ({c: np.concatenate(v) for c, v in reg.items()},
            {c: np.concatenate(v) for c, v in pdf.items()})


def score(regs_log, p_def_map, Yva, va_def, extra=""):
    grid = np.arange(0.02, 0.71, 0.02)
    th, dec = {}, {}
    for c in TARGETS:
        y = Yva[c].to_numpy()
        m = ~np.isnan(y)
        p = p_def_map[c]
        best_t, best_a = 0.5, -1.0
        for t in grid:
            pred = np.where(p >= t, DEFAULT_VALUES[c], np.power(10.0, regs_log[c]))
            a = pointwise(c, pred[m], y[m]).mean()
            if a > best_a:
                best_a, best_t = a, float(t)
        th[c] = best_t
        pred = np.where(p >= best_t, DEFAULT_VALUES[c], np.power(10.0, regs_log[c]))
        dec[c] = decompose(c, pred, y, va_def[c].to_numpy() & m)
    total = sum(WEIGHTS[c] * dec[c]["acc"] for c in TARGETS) * 100
    return total, dec, th


def main(tags, weights_grid, decision="tree"):
    wells = load_wells("./train", has_labels=True)
    ids = [w["logId"].iloc[0] for w in wells]
    tr = [w for w in wells if w["logId"].iloc[0] in set(ids[:72])]
    va = [w for w in wells if w["logId"].iloc[0] in set(ids[72:])]
    tr_df, va_df = pd.concat(tr, ignore_index=True), pd.concat(va, ignore_index=True)
    Xva = make_features(va_df, context=True, priors=True)
    Yva = make_targets(va_df)
    va_def = default_mask(Yva)

    cache_path = os.path.join(RUNS, "screen_tree_cache.joblib")
    if os.path.exists(cache_path):
        cch = joblib.load(cache_path)
        if list(cch["va_ids"]) == ids[72:]:
            p_def_tree, reg_tree = cch["p_def"], cch["reg"]
            print("loaded tree cache")
        else:
            cch = None
    else:
        cch = None
    if cch is None:
        model, _ = build_tree(tr_df, weight_tol=True)
        p_def_tree = {c: model.classifiers[c].predict_proba(Xva)[:, 1] for c in TARGETS}
        reg_tree = {c: np.log10(np.maximum(inverse_transform_target(
            c, pd.Series(model.regressors[c].predict(Xva), index=Xva.index)).to_numpy(), EPS)) for c in TARGETS}
        joblib.dump({"va_ids": ids[72:], "p_def": p_def_tree, "reg": reg_tree}, cache_path)
    print(f"tree_only(weight_tol): ", end="", flush=True)
    t, d, _ = score(reg_tree, p_def_tree, Yva, va_def)
    print(f"{t:.3f} real=({d['POR']['acc_real']:.4f},{d['PERM']['acc_real']:.4f},{d['SW']['acc_real']:.4f})")

    arts = {}
    for tag in tags:
        ap = os.path.join(RUNS, f"tcn_{tag}.joblib")
        if not os.path.exists(ap):
            print(f"[skip] {tag} 缺失 {ap}")
            continue
        arts[tag] = joblib.load(ap)
        print(f"loaded {tag}")

    # 各 seed 单独 + 集成平均
    regs_by_tag, p_by_tag = {}, {}
    for tag, art in arts.items():
        r, p = tcn_outputs(va, art)
        regs_by_tag[tag], p_by_tag[tag] = r, p
        t, d, _ = score(r, p_def_tree, Yva, va_def)
        print(f"tcn {tag:12s}: total={t:.3f} real=({d['POR']['acc_real']:.4f},{d['PERM']['acc_real']:.4f},{d['SW']['acc_real']:.4f})")

    ens = {c: np.mean([regs_by_tag[t2][c] for t2 in regs_by_tag], axis=0) for c in TARGETS}
    p_ens = {c: np.mean([p_by_tag[t2][c] for t2 in p_by_tag], axis=0) for c in TARGETS}
    print("--- blends (decision=tree) ---")
    best = None
    for wmap in weights_grid:
        regs = {c: (1 - wmap.get("tcn", 0.7)) * reg_tree[c] + wmap.get("tcn", 0.7) * ens[c] for c in TARGETS}
        t, d, th = score(regs, p_def_tree, Yva, va_def)
        label = f"tree{1 - wmap.get('tcn', 0.7):.2f}+ens{wmap.get('tcn', 0.7):.2f}"
        print(f"{label:18s}: total={t:.3f} real=({d['POR']['acc_real']:.4f},{d['PERM']['acc_real']:.4f},{d['SW']['acc_real']:.4f}) def=({d['POR']['acc_def']:.4f},{d['PERM']['acc_def']:.4f},{d['SW']['acc_def']:.4f})")
        if best is None or t > best[0]:
            best = (t, label, wmap)
    print("best blend:", best)

    wmap = best[2]
    regs = {c: (1 - wmap.get("tcn", 0.7)) * reg_tree[c] + wmap.get("tcn", 0.7) * ens[c] for c in TARGETS}
    if decision == "mean_prob":
        p_mix = {c: 0.5 * p_def_tree[c] + 0.5 * p_ens[c] for c in TARGETS}
        t, d, th = score(regs, p_mix, Yva, va_def)
        print(f"mean_prob decision: total={t:.3f} real=({d['POR']['acc_real']:.4f},{d['PERM']['acc_real']:.4f},{d['SW']['acc_real']:.4f}) th={th}")
        for a in (0.2, 0.3, 0.4, 0.6, 0.7, 0.8):
            pm = {c: (1 - a) * p_def_tree[c] + a * p_ens[c] for c in TARGETS}
            t2, d2, th2 = score(regs, pm, Yva, va_def)
            print(f"  mix a={a}: total={t2:.3f} real=({d2['POR']['acc_real']:.4f},{d2['PERM']['acc_real']:.4f},{d2['SW']['acc_real']:.4f})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default="T6_long,T8_seed43,T9_seed44")
    ap.add_argument("--wd", default="0.5,0.6,0.7,0.8,0.9")
    ap.add_argument("--decision", default="tree")
    a = ap.parse_args()
    tags = [t.strip() for t in a.tags.split(",") if t.strip()]
    ws = [float(x) for x in a.wd.split(",")]
    main(tags, [{"tcn": w} for w in ws], a.decision)
