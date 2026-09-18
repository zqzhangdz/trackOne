"""screen 切分最终混合评估：树回归 / T3(pass1) / T5(pass2) 的 log10 空间加权融合."""
from __future__ import annotations

import itertools
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.feature import default_mask, inverse_transform_target, load_wells, make_features, make_targets
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


def tcn_regs(wells_df_ids, va_df, va_wells, art, channels=None):
    medians, stds = art["medians"], art["stds"]
    regs = {c: np.zeros(len(va_df)) for c in TARGETS}
    ids = va_df["logId"].to_numpy()
    for w in va_wells:
        Xw = make_features(w, context=False, priors=True).fillna(medians)
        Xw = ((Xw - stds.replace(0, 1.0)) / stds.replace(0, 1.0)).to_numpy(dtype=np.float32)
        if channels is not None:
            ch = channels[w["logId"].iloc[0]]
            Xw = np.concatenate([Xw, ch], axis=1).astype(np.float32)
        o = numpy_forward(Xw, art["sd"], art["dilations"])
        m = ids == w["logId"].iloc[0]
        for c in TARGETS:
            regs[c][m] = o[c][:, 1]
    return regs


def main(t3_tag="T3_sel80", t5_tag="T5_pass2", weight_tol=False):
    wells = load_wells("./train", has_labels=True)
    ids = [w["logId"].iloc[0] for w in wells]
    tr = [w for w in wells if w["logId"].iloc[0] in set(ids[:72])]
    va = [w for w in wells if w["logId"].iloc[0] in set(ids[72:])]
    tr_df, va_df = pd.concat(tr, ignore_index=True), pd.concat(va, ignore_index=True)
    Xtr = make_features(tr_df, context=True, priors=True)
    Xva = make_features(va_df, context=True, priors=True)
    Ytr, Yva = make_targets(tr_df), make_targets(va_df)
    model = TwoStageModel(monotonic=True).fit(Xtr, Ytr, default_mask(Ytr))
    if weight_tol:
        for c in TARGETS:
            valid = Ytr[c].notna().to_numpy()
            y = Ytr.loc[valid, c]
            def_ = default_mask(Ytr).loc[valid, c].astype(int).to_numpy()
            real = def_ == 0
            yv = y.to_numpy()[real]
            w = np.abs(np.log10(np.maximum(yv, EPS))) if c == "PERM" else np.abs(yv)
            w = np.clip(w, np.percentile(w, 5), np.percentile(w, 95))
            w = w / w.mean()
            from src.feature import transform_target as _tt
            model.regressors[c].fit(Xtr.loc[valid][real], _tt(c, y[real]), sample_weight=w)
    p_def = {c: model.classifiers[c].predict_proba(Xva)[:, 1] for c in TARGETS}
    reg_tree = {c: np.log10(np.maximum(inverse_transform_target(
        c, pd.Series(model.regressors[c].predict(Xva), index=Xva.index)).to_numpy(), EPS)) for c in TARGETS}

    art3 = joblib.load(os.path.join(RUNS, f"tcn_{t3_tag}.joblib"))
    reg3 = tcn_regs(None, va_df, va, art3)
    reg5 = None
    p5 = os.path.join(RUNS, f"tcn_{t5_tag}.joblib")
    if os.path.exists(p5):
        art5 = joblib.load(p5)
        ch = joblib.load(os.path.join(RUNS, f"pass1_val_{art5['pass1_tag']}.joblib"))
        ch = {k: np.stack([v[c] for c in TARGETS], axis=1) for k, v in ch.items()}
        reg5 = tcn_regs(None, va_df, va, art5, channels=ch)

    sources = {"tree": reg_tree, "t3": reg3}
    if reg5 is not None:
        sources["t5"] = reg5

    grid = np.arange(0.05, 0.96, 0.05)

    def score_regs(regs):
        th, dec = {}, {}
        for c in TARGETS:
            y = Yva[c].to_numpy()
            m = ~np.isnan(y)
            best_t, best_a = 0.5, -1.0
            for t in grid:
                pred = np.where(p_def[c] >= t, DEFAULT_VALUES[c], np.power(10.0, regs[c]))
                a = pointwise(c, pred[m], y[m]).mean()
                if a > best_a:
                    best_a, best_t = a, float(t)
            th[c] = best_t
            pred = np.where(p_def[c] >= best_t, DEFAULT_VALUES[c], np.power(10.0, regs[c]))
            isdef = default_mask(Yva)[c].to_numpy() & m
            dec[c] = decompose(c, pred, y, isdef)
        return sum(WEIGHTS[c] * dec[c]["acc"] for c in TARGETS) * 100, dec, th

    for name, regs in sources.items():
        tot, dec, th = score_regs(regs)
        print(f"{name:5s}: total={tot:.3f} | " + " ".join(f"{c} real={dec[c]['acc_real']:.4f}" for c in TARGETS))

    names = list(sources.keys())
    if len(names) >= 2:
        weights_grid = []
        for combo in itertools.combinations(names, 2):
            for w in (0.2, 0.3, 0.4, 0.5, 0.7, 0.8):
                weights_grid.append(({combo[0]: 1 - w, combo[1]: w}, f"{combo[0]}{1 - w:.1f}+{combo[1]}{w:.1f}"))
        if len(names) == 3:
            for w3 in ((0.4, 0.3, 0.3), (0.34, 0.33, 0.33), (0.2, 0.4, 0.4), (0.5, 0.25, 0.25)):
                weights_grid.append((dict(zip(names, w3)), "+".join(f"{n}{w:.2f}" for n, w in zip(names, w3))))
        results = []
        for wmap, label in weights_grid:
            regs = {c: sum(wmap[n] * sources[n][c] for n in wmap) for c in TARGETS}
            tot, dec, _ = score_regs(regs)
            results.append((tot, label, dec))
        results.sort(reverse=True, key=lambda r: r[0])
        for tot, label, dec in results[:6]:
            print(f"blend {label:22s}: total={tot:.3f} | " + " ".join(f"{c} real={dec[c]['acc_real']:.4f}" for c in TARGETS))


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--t3_tag", default="T3_sel80")
    ap.add_argument("--t5_tag", default="T5_pass2")
    ap.add_argument("--weight_tol", type=int, default=0)
    a = ap.parse_args()
    main(a.t3_tag, a.t5_tag, bool(a.weight_tol))
