"""统一 CV 评估入口：screen(72/8 切分) 或 5 折 GroupKFold，输出分数分解并追加到 results.csv.

用法:
  python experiments/run_cv.py --mode screen --tag A_baseline
  python experiments/run_cv.py --mode cv --folds 5 --tag B_ctx --context 1 --priors 1
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.feature import default_mask, load_wells, make_features, make_targets
from src.model import TwoStageModel, predict as model_predict
from src.utils import DEFAULT_VALUES, TARGETS, TOLERANCE, WEIGHTS, EPS

RESULTS_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.csv")
THRESHOLD_GRID = np.arange(0.05, 0.96, 0.05)


def pointwise(c: str, pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    p = np.asarray(pred, float)
    t = np.asarray(true, float)
    if c == "PERM":
        p = np.log10(np.maximum(p, EPS))
        t = np.log10(np.maximum(t, EPS))
    denom = TOLERANCE[c] * (np.abs(t) + EPS)
    return np.maximum(0.0, 1.0 - np.abs(p - t) / denom)


def decompose(c: str, pred: np.ndarray, y: np.ndarray, isdef: np.ndarray) -> dict:
    valid = ~np.isnan(y)
    d = valid & isdef
    r = valid & ~isdef
    return {
        "acc": pointwise(c, pred[valid], y[valid]).mean(),
        "frac_def": d.sum() / max(1, valid.sum()),
        "acc_def": pointwise(c, pred[d], y[d]).mean() if d.sum() else float("nan"),
        "acc_real": pointwise(c, pred[r], y[r]).mean() if r.sum() else float("nan"),
    }


def tune_and_eval(model: TwoStageModel, X: pd.DataFrame, Y: pd.DataFrame, isdef: pd.DataFrame):
    p_def = {c: model.classifiers[c].predict_proba(X)[:, 1] for c in TARGETS}
    mode = model.target_transform
    from src.feature import inverse_transform_target

    reg_phys = {
        c: inverse_transform_target(
            c, pd.Series(model.regressors[c].predict(X), index=X.index), mode
        ).to_numpy()
        for c in TARGETS
    }
    th = {}
    dec = {}
    for c in TARGETS:
        y = Y[c].to_numpy()
        m = ~np.isnan(y)
        best_t, best_a = 0.5, -1.0
        for t in THRESHOLD_GRID:
            pred = np.where(p_def[c] >= t, DEFAULT_VALUES[c], reg_phys[c])
            a = pointwise(c, pred[m], y[m]).mean()
            if a > best_a:
                best_a, best_t = a, float(t)
        th[c] = best_t
        pred = np.where(p_def[c] >= best_t, DEFAULT_VALUES[c], reg_phys[c])
        dec[c] = decompose(c, pred, y, isdef[c].to_numpy())
    total = sum(WEIGHTS[c] * dec[c]["acc"] for c in TARGETS) * 100.0
    return total, dec, th


def run_fold(X, Y, isdef, tr_idx, va_idx, args):
    model = TwoStageModel(
        target_transform=args.ttransform,
        sample_weight=bool(args.weights),
        monotonic=bool(args.mono),
    ).fit(X.iloc[tr_idx], Y.iloc[tr_idx], isdef.iloc[tr_idx])
    total, dec, th = tune_and_eval(model, X.iloc[va_idx], Y.iloc[va_idx], isdef.iloc[va_idx])
    return total, dec, th


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="./train")
    ap.add_argument("--mode", choices=["screen", "cv"], default="screen")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--context", type=int, default=0)
    ap.add_argument("--priors", type=int, default=0)
    ap.add_argument("--ttransform", choices=["perm_log", "all_log"], default="perm_log")
    ap.add_argument("--weights", type=int, default=0)
    ap.add_argument("--mono", type=int, default=0)
    args = ap.parse_args()

    t0 = time.time()
    wells = load_wells(args.data_dir, has_labels=True)
    df = pd.concat(wells, ignore_index=True)
    X = make_features(df, context=bool(args.context), priors=bool(args.priors))
    Y = make_targets(df)
    isdef = default_mask(Y)
    groups = df["logId"].to_numpy()
    print(f"[{args.tag}] wells={len(wells)} rows={len(df):,} features={X.shape[1]}")

    if args.mode == "screen":
        ids = pd.unique(groups)
        tr_mask = np.isin(groups, ids[:72])
        splits = [(np.where(tr_mask)[0], np.where(~tr_mask)[0])]
    else:
        gkf = GroupKFold(n_splits=args.folds)
        splits = list(gkf.split(X, Y, groups))

    totals, decs, ths = [], [], []
    for fi, (tr_idx, va_idx) in enumerate(splits, start=1):
        total, dec, th = run_fold(X, Y, isdef, tr_idx, va_idx, args)
        totals.append(total)
        decs.append(dec)
        ths.append(th)
        line = f"  fold{fi}: total={total:.3f} " + " ".join(
            f"{c}={dec[c]['acc']:.4f}(def {dec[c]['acc_def']:.3f}/real {dec[c]['acc_real']:.3f})" for c in TARGETS
        )
        print(line, flush=True)

    mean_dec = {
        c: {k: float(np.nanmean([d[c][k] for d in decs])) for k in decs[0][c]} for c in TARGETS
    }
    mean_total = float(np.mean(totals))
    std_total = float(np.std(totals))
    print(
        f"[{args.tag}] MEAN total={mean_total:.3f} std={std_total:.3f} | "
        + " ".join(f"{c}: acc={mean_dec[c]['acc']:.4f} real={mean_dec[c]['acc_real']:.4f}" for c in TARGETS)
        + f" | {time.time() - t0:.0f}s"
    )

    row = {
        "tag": args.tag, "mode": args.mode, "folds": len(splits),
        "context": args.context, "priors": args.priors, "ttransform": args.ttransform,
        "weights": args.weights, "mono": args.mono,
        "total": round(mean_total, 3), "std": round(std_total, 3),
        **{f"{c}_acc": round(mean_dec[c]["acc"], 4) for c in TARGETS},
        **{f"{c}_real": round(mean_dec[c]["acc_real"], 4) for c in TARGETS},
        **{f"{c}_def": round(mean_dec[c]["acc_def"], 4) for c in TARGETS},
    }
    new = not os.path.exists(RESULTS_CSV)
    with open(RESULTS_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new:
            w.writeheader()
        w.writerow(row)
    print(f"appended to {RESULTS_CSV}")


if __name__ == "__main__":
    main()
