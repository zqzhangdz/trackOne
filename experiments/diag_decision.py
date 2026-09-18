"""决策损失分解 + 联合分段决策实验（screen 验证井）.

不导入 torch。目标：量化默认段/真实段误判的丢分结构，测试
(a) 逐目标阈值（基线）(b) 三目标联合 logit 决策 (c) 分类概率融合（tree+TCN）。
"""
from __future__ import annotations
import os, sys, warnings
warnings.filterwarnings("ignore")
import joblib, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.feature import (default_mask, inverse_transform_target, load_wells, make_features,
                         make_targets, transform_target)
from src.model import TwoStageModel
from src.utils import DEFAULT_VALUES, EPS, TARGETS, TOLERANCE, WEIGHTS
from solution.tcn_model import numpy_forward
RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")

def pointwise(c, pred, true):
    p = np.asarray(pred, float); t = np.asarray(true, float)
    if c == "PERM":
        p = np.log10(np.maximum(p, EPS)); t = np.log10(np.maximum(t, EPS))
    return np.maximum(0.0, 1.0 - np.abs(p - t) / (TOLERANCE[c] * (np.abs(t) + EPS)))

def build():
    wells = load_wells("./train", has_labels=True)
    ids = [w["logId"].iloc[0] for w in wells]
    tr = [w for w in wells if w["logId"].iloc[0] in set(ids[:72])]
    va = [w for w in wells if w["logId"].iloc[0] in set(ids[72:])]
    tr_df, va_df = pd.concat(tr, ignore_index=True), pd.concat(va, ignore_index=True)
    Xtr, Xva = make_features(tr_df, context=True, priors=True), make_features(va_df, context=True, priors=True)
    Ytr, Yva = make_targets(tr_df), make_targets(va_df)
    m = TwoStageModel(monotonic=True).fit(Xtr, Ytr, default_mask(Ytr))
    for c in TARGETS:
        valid = Ytr[c].notna().to_numpy(); y = Ytr.loc[valid, c]
        real = default_mask(Ytr).loc[valid, c].astype(int).to_numpy() == 0
        yv = y.to_numpy()[real]
        w = np.abs(np.log10(np.maximum(yv, EPS))) if c == "PERM" else np.abs(yv)
        w = np.clip(w, np.percentile(w, 5), np.percentile(w, 95)); w /= w.mean()
        m.regressors[c].fit(Xtr.loc[valid][real], transform_target(c, y[real]), sample_weight=w)
    return va, va_df, Xva, Yva, m

def main():
    va, va_df, Xva, Yva, model = build()
    p_def = {c: model.classifiers[c].predict_proba(Xva)[:, 1] for c in TARGETS}
    reg = {c: np.log10(np.maximum(inverse_transform_target(
        c, pd.Series(model.regressors[c].predict(Xva), index=Xva.index)).to_numpy(), EPS)) for c in TARGETS}
    # TCN p_def 融合
    art = joblib.load(os.path.join(RUNS, "tcn_T6_long.joblib"))
    p_tcn = {}
    for w in va:
        Xw = make_features(w, context=False, priors=True).fillna(art["medians"])
        Xw = ((Xw - art["stds"].replace(0, 1.0)) / art["stds"].replace(0, 1.0)).to_numpy(np.float32)
        o = numpy_forward(Xw, art["sd"], art["dilations"])
        for c in TARGETS:
            p_tcn.setdefault(c, []).append(1.0 / (1.0 + np.exp(-o[c][:, 0])))
    p_tcn = {c: np.concatenate(v) for c, v in p_tcn.items()}

    grid = np.arange(0.02, 0.71, 0.02)
    def eval_decision(pmap, joint=False):
        th, dec = {}, {}
        logits = {c: np.log(np.clip(pmap[c], 1e-9, 1-1e-9) / (1 - np.clip(pmap[c], 1e-9, 1-1e-9))) for c in TARGETS}
        if joint:
            lj = sum(logits[c] for c in TARGETS) / 3.0
            best_t, best_tot = 0.5, -1
            for t in grid:
                tot = 0.0
                for c in TARGETS:
                    y = Yva[c].to_numpy(); m = ~np.isnan(y)
                    pred = np.where(lj >= t, DEFAULT_VALUES[c], np.power(10.0, reg[c]))
                    tot += WEIGHTS[c] * pointwise(c, pred[m], y[m]).mean()
                if tot > best_tot: best_tot, best_t = tot, float(t)
            for c in TARGETS:
                th[c] = best_t
                y = Yva[c].to_numpy(); m = ~np.isnan(y)
                pred = np.where(lj >= best_t, DEFAULT_VALUES[c], np.power(10.0, reg[c]))
                isdef = default_mask(Yva)[c].to_numpy() & m
                dec[c] = {"acc": pointwise(c, pred[m], y[m]).mean(),
                          "acc_def": pointwise(c, pred[isdef], y[isdef]).mean(),
                          "acc_real": pointwise(c, pred[m & ~isdef], y[m & ~isdef]).mean()}
        else:
            for c in TARGETS:
                y = Yva[c].to_numpy(); m = ~np.isnan(y)
                best_t, best_a = 0.5, -1
                for t in grid:
                    pred = np.where(pmap[c] >= t, DEFAULT_VALUES[c], np.power(10.0, reg[c]))
                    a = pointwise(c, pred[m], y[m]).mean()
                    if a > best_a: best_a, best_t = a, float(t)
                th[c] = best_t
                pred = np.where(pmap[c] >= best_t, DEFAULT_VALUES[c], np.power(10.0, reg[c]))
                isdef = default_mask(Yva)[c].to_numpy() & m
                dec[c] = {"acc": pointwise(c, pred[m], y[m]).mean(),
                          "acc_def": pointwise(c, pred[isdef], y[isdef]).mean(),
                          "acc_real": pointwise(c, pred[m & ~isdef], y[m & ~isdef]).mean()}
        return sum(WEIGHTS[c]*dec[c]["acc"] for c in TARGETS)*100, dec, th

    for name, pm in [("tree", p_def), ("tcn", p_tcn), ("mean", {c: .5*p_def[c]+.5*p_tcn[c] for c in TARGETS})]:
        t, d, th = eval_decision(pm, False)
        print(f"{name:5s} per-target: {t:.3f} real=({d['POR']['acc_real']:.4f},{d['PERM']['acc_real']:.4f},{d['SW']['acc_real']:.4f}) def=({d['POR']['acc_def']:.4f},{d['PERM']['acc_def']:.4f},{d['SW']['acc_def']:.4f}) th={ {k: round(v,2) for k,v in th.items()} }")
        t, d, th = eval_decision(pm, True)
        print(f"{name:5s} joint     : {t:.3f} real=({d['POR']['acc_real']:.4f},{d['PERM']['acc_real']:.4f},{d['SW']['acc_real']:.4f}) def=({d['POR']['acc_def']:.4f},{d['PERM']['acc_def']:.4f},{d['SW']['acc_def']:.4f}) th={ {k: round(v,2) for k,v in th.items()} }")

    # 负面丢分分解：真默认被判真 vs 真真实被判默认
    t, d, th = eval_decision(p_def, False)
    for c in TARGETS:
        y = Yva[c].to_numpy(); m = ~np.isnan(y)
        isdef = default_mask(Yva)[c].to_numpy() & m
        pred_def = p_def[c] >= th[c]
        # 实际给分
        print(f"[{c}] 真默认点数={isdef.sum()} 真真实点数={(m & ~isdef).sum()}")
        print(f"   默认->预测默认 acc={pointwise(c, np.array([DEFAULT_VALUES[c]]), y[isdef & pred_def]).mean() if (isdef&pred_def).sum() else float('nan'):.3f} n={int((isdef&pred_def).sum())}")
        bad = isdef & ~pred_def
        print(f"   真默认->判真实 acc={pointwise(c, np.power(10.0, reg[c])[bad], y[bad]).mean() if bad.sum() else float('nan'):.3f} n={int(bad.sum())}")
        bad2 = (m & ~isdef) & pred_def
        print(f"   真真实->判默认 acc={pointwise(c, np.array([DEFAULT_VALUES[c]]), y[bad2]).mean() if bad2.sum() else float('nan'):.3f} n={int(bad2.sum())}")

if __name__ == "__main__":
    main()
