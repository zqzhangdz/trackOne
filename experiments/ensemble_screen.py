"""screen 切分上的混合实验：树分类器决策 × {树回归 / TCN回归 / 两者log10平均}."""
from __future__ import annotations

import os
import sys
import warnings

warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.feature import default_mask, load_wells, make_features, make_targets
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
    denom = TOLERANCE[c] * (np.abs(t) + EPS)
    return np.maximum(0.0, 1.0 - np.abs(p - t) / denom)


def main(tcn_tag: str = "T2_full", w_tcn: float = 0.5) -> None:
    wells = load_wells("./train", has_labels=True)
    ids = [w["logId"].iloc[0] for w in wells]
    tr = [w for w in wells if w["logId"].iloc[0] in set(ids[:72])]
    va = [w for w in wells if w["logId"].iloc[0] in set(ids[72:])]
    tr_df = pd.concat(tr, ignore_index=True)
    va_df = pd.concat(va, ignore_index=True)
    Xtr = make_features(tr_df, context=True, priors=True)
    Xva = make_features(va_df, context=True, priors=True)
    Ytr, Yva = make_targets(tr_df), make_targets(va_df)
    dtr = default_mask(Ytr)

    model = TwoStageModel(monotonic=True).fit(Xtr, Ytr, dtr)
    p_def = {c: model.classifiers[c].predict_proba(Xva)[:, 1] for c in TARGETS}
    from src.feature import inverse_transform_target, transform_target

    reg_tree_log = {
        c: np.log10(np.maximum(inverse_transform_target(
            c, pd.Series(model.regressors[c].predict(Xva), index=Xva.index)).to_numpy(), EPS))
        for c in TARGETS
    }

    art = joblib.load(os.path.join(RUNS, f"tcn_{tcn_tag}.joblib"))
    medians, stds = art["medians"], art["stds"]
    reg_tcn_log, p_def_tcn = {}, {}
    va_ids = va_df["logId"].to_numpy()
    for w in va:
        lid = w["logId"].iloc[0]
        Xw = make_features(w, context=False, priors=True).fillna(medians)
        Xw = ((Xw - stds.replace(0, 1.0)) / stds.replace(0, 1.0)).to_numpy(dtype=np.float32)
        o = numpy_forward(Xw, art["sd"], art["dilations"])
        m = va_ids == lid
        for c in TARGETS:
            reg_tcn_log.setdefault(c, np.zeros(len(va_df)))[m] = o[c][:, 1]
            p_def_tcn.setdefault(c, np.zeros(len(va_df)))[m] = 1.0 / (1.0 + np.exp(-o[c][:, 0]))

    variants = {
        "tree_only": {c: reg_tree_log[c] for c in TARGETS},
        "tcn_only": {c: reg_tcn_log[c] for c in TARGETS},
        "blend50": {c: (1 - w_tcn) * reg_tree_log[c] + w_tcn * reg_tcn_log[c] for c in TARGETS},
    }
    grid = np.arange(0.05, 0.96, 0.05)
    for name, regs in variants.items():
        th = {}
        dec = {}
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
            valid, isdef = m, default_mask(Yva)[c].to_numpy() & m
            real = m & ~isdef
            dec[c] = {
                "acc": pointwise(c, pred[m], y[m]).mean(),
                "acc_def": pointwise(c, pred[isdef], y[isdef]).mean(),
                "acc_real": pointwise(c, pred[real], y[real]).mean(),
            }
        total = sum(WEIGHTS[c] * dec[c]["acc"] for c in TARGETS) * 100
        print(
            f"{name}: total={total:.3f} | "
            + " ".join(f"{c} acc={dec[c]['acc']:.4f} real={dec[c]['acc_real']:.4f}" for c in TARGETS)
            + f" | th={ {k: round(v, 2) for k, v in th.items()} }"
        )


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--tcn_tag", default="T2_full")
    ap.add_argument("--w_tcn", type=float, default=0.5)
    a = ap.parse_args()
    main(a.tcn_tag, a.w_tcn)
