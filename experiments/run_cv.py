"""统一 CV 评估入口：screen(72/8 切分) 或 5 折 GroupKFold，输出分数分解并追加到 results.csv.

用法:
  python experiments/run_cv.py --mode screen --tag A_baseline
  python experiments/run_cv.py --mode cv --folds 5 --tag B_ctx --context 1 --priors 1
  python experiments/run_cv.py --mode screen --tag G_seg --context 1 --priors 1 --seg viterbi
"""
from __future__ import annotations

import argparse
import csv
import warnings

warnings.filterwarnings("ignore")
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.feature import default_mask, inverse_transform_target, load_wells, make_features, make_targets
from src.model import TwoStageModel
from src.utils import DEFAULT_VALUES, TARGETS, TOLERANCE, WEIGHTS, EPS

RESULTS_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.csv")
THRESHOLD_GRID = np.arange(0.05, 0.96, 0.05)
BIAS_GRID = np.arange(-2.0, 2.01, 0.5)


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


def _runs_mean_run_length(mask: np.ndarray) -> float:
    diffs = np.diff(mask.astype(np.int8))
    starts = np.count_nonzero(diffs == 1) + (1 if mask.size and mask[0] else 0)
    return float(mask.sum() / max(1, starts))


def _viterbi(ll: np.ndarray, bias: float, log_stay: float, log_switch: float) -> np.ndarray:
    """2 状态(0=真实,1=默认) Viterbi. ll = logit(p_default)."""
    n = ll.size
    v0 = 0.0
    v1 = ll[0] + bias
    ptr = np.zeros((n, 2), dtype=np.int8)
    vs = np.zeros((n, 2))
    vs[0] = (v0, v1)
    for t in range(1, n):
        e = ll[t] + bias
        s00, s01 = vs[t - 1, 0] + log_stay, vs[t - 1, 1] + log_switch
        s10, s11 = vs[t - 1, 0] + log_switch, vs[t - 1, 1] + log_stay
        vs[t, 0] = max(s00, s01)
        ptr[t, 0] = 1 if s01 > s00 else 0
        vs[t, 1] = max(s10, s11) + e
        ptr[t, 1] = 1 if s11 > s10 else 0
    path = np.zeros(n, dtype=np.int8)
    path[-1] = 1 if vs[-1, 1] > vs[-1, 0] else 0
    for t in range(n - 1, 0, -1):
        path[t - 1] = ptr[t, path[t]]
    return path.astype(bool)


def smooth_mask(mask: np.ndarray, groups: np.ndarray, seg: str, win: int,
                logit_p: np.ndarray, log_stay: float, log_switch: float, bias: float) -> np.ndarray:
    if seg == "none":
        return mask
    out = mask.copy()
    for lid in pd.unique(groups):
        m = groups == lid
        if seg == "median":
            s = pd.Series(mask[m].astype(np.int8))
            out[m] = (s.rolling(win, center=True, min_periods=1).mean() >= 0.5).to_numpy()
        else:
            ll = np.clip(logit_p[m], -20, 20)
            out[m] = _viterbi(ll, bias, log_stay, log_switch)
    return out


def tune_and_eval(model, X, Y, isdef, groups, seg, seg_win, p_stay):
    p_def = {c: model.classifiers[c].predict_proba(X)[:, 1] for c in TARGETS}
    mode = model.target_transform
    reg_phys = {
        c: inverse_transform_target(c, pd.Series(model.regressors[c].predict(X), index=X.index), mode).to_numpy()
        for c in TARGETS
    }
    th = {}
    dec = {}
    for c in TARGETS:
        y = Y[c].to_numpy()
        m = ~np.isnan(y)
        log_stay = float(np.log(p_stay[c]))
        log_switch = float(np.log(1.0 - p_stay[c]))
        logit_p = np.log(np.clip(p_def[c], 1e-6, 1 - 1e-6) / np.clip(1 - p_def[c], 1e-6, 1 - 1e-6))
        if seg == "viterbi":
            best_b, best_a = 0.0, -1.0
            for b in BIAS_GRID:
                sm = smooth_mask(np.zeros_like(m), groups, seg, seg_win, logit_p, log_stay, log_switch, b)
                pred = np.where(sm, DEFAULT_VALUES[c], reg_phys[c])
                a = pointwise(c, pred[m], y[m]).mean()
                if a > best_a:
                    best_a, best_b = a, float(b)
            sm = smooth_mask(np.zeros_like(m), groups, seg, seg_win, logit_p, log_stay, log_switch, best_b)
            th[c] = best_b
        else:
            best_t, best_a = 0.5, -1.0
            for t in THRESHOLD_GRID:
                raw = p_def[c] >= t
                sm = smooth_mask(raw, groups, seg, seg_win, logit_p, log_stay, log_switch, 0.0)
                pred = np.where(sm, DEFAULT_VALUES[c], reg_phys[c])
                a = pointwise(c, pred[m], y[m]).mean()
                if a > best_a:
                    best_a, best_t = a, float(t)
            sm = smooth_mask(p_def[c] >= best_t, groups, seg, seg_win, logit_p, log_stay, log_switch, 0.0)
            th[c] = best_t
        pred = np.where(sm, DEFAULT_VALUES[c], reg_phys[c])
        dec[c] = decompose(c, pred, y, isdef[c].to_numpy())
    total = sum(WEIGHTS[c] * dec[c]["acc"] for c in TARGETS) * 100.0
    return total, dec, th


def run_fold(X, Y, isdef, tr_idx, va_idx, args):
    model = TwoStageModel(
        target_transform=args.ttransform,
        sample_weight=bool(args.weights),
        monotonic=bool(args.mono),
    ).fit(X.iloc[tr_idx], Y.iloc[tr_idx], isdef.iloc[tr_idx])
    p_stay = {}
    for c in TARGETS:
        seq = isdef[c].to_numpy()[tr_idx]
        p_stay[c] = min(0.9995, max(0.9, 1.0 - 1.0 / max(2.0, _runs_mean_run_length(seq))))
    total, dec, th = tune_and_eval(
        model, X.iloc[va_idx], Y.iloc[va_idx], isdef.iloc[va_idx],
        groups=GROUPS[va_idx],
        seg=args.seg, seg_win=args.seg_win, p_stay=p_stay,
    )
    return total, dec, th


GROUPS = None


def main() -> None:
    global GROUPS
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
    ap.add_argument("--seg", choices=["none", "median", "viterbi"], default="none")
    ap.add_argument("--seg_win", type=int, default=15)
    args = ap.parse_args()

    t0 = time.time()
    wells = load_wells(args.data_dir, has_labels=True)
    df = pd.concat(wells, ignore_index=True)
    X = make_features(df, context=bool(args.context), priors=bool(args.priors))
    Y = make_targets(df)
    isdef = default_mask(Y)
    GROUPS = df["logId"].to_numpy()
    print(f"[{args.tag}] wells={len(wells)} rows={len(df):,} features={X.shape[1]} seg={args.seg}")

    if args.mode == "screen":
        ids = pd.unique(GROUPS)
        tr_mask = np.isin(GROUPS, ids[:72])
        splits = [(np.where(tr_mask)[0], np.where(~tr_mask)[0])]
    else:
        gkf = GroupKFold(n_splits=args.folds)
        splits = list(gkf.split(X, Y, GROUPS))

    totals, decs, ths = [], [], []
    for fi, (tr_idx, va_idx) in enumerate(splits, start=1):
        total, dec, th = run_fold(X, Y, isdef, tr_idx, va_idx, args)
        totals.append(total)
        decs.append(dec)
        ths.append(th)
        print(
            f"  fold{fi}: total={total:.3f} " + " ".join(
                f"{c}={dec[c]['acc']:.4f}(def {dec[c]['acc_def']:.3f}/real {dec[c]['acc_real']:.3f})" for c in TARGETS
            ),
            flush=True,
        )

    mean_dec = {c: {k: float(np.nanmean([d[c][k] for d in decs])) for k in decs[0][c]} for c in TARGETS}
    mean_total = float(np.mean(totals))
    std_total = float(np.std(totals))
    print(
        f"[{args.tag}] MEAN total={mean_total:.3f} std={std_total:.3f} | "
        + " ".join(f"{c}: acc={mean_dec[c]['acc']:.4f} real={mean_dec[c]['acc_real']:.4f} def={mean_dec[c]['acc_def']:.4f}" for c in TARGETS)
        + f" | th={ {k: round(v, 2) for k, v in ths[-1].items()} } | {time.time() - t0:.0f}s"
    )

    row = {
        "tag": args.tag, "mode": args.mode, "folds": len(splits),
        "context": args.context, "priors": args.priors, "ttransform": args.ttransform,
        "weights": args.weights, "mono": args.mono, "seg": args.seg, "seg_win": args.seg_win,
        "total": round(mean_total, 3), "std": round(std_total, 3),
        **{f"{c}_acc": round(mean_dec[c]["acc"], 4) for c in TARGETS},
        **{f"{c}_real": round(mean_dec[c]["acc_real"], 4) for c in TARGETS},
        **{f"{c}_def": round(mean_dec[c]["acc_def"], 4) for c in TARGETS},
    }
    fieldnames = list(row.keys())
    if os.path.exists(RESULTS_CSV):
        with open(RESULTS_CSV, encoding="utf-8") as f:
            old_header = next(csv.reader(f), None)
        if old_header:
            fieldnames = list(old_header) + [k for k in row.keys() if k not in old_header]
    with open(RESULTS_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, restval="")
        if os.path.getsize(RESULTS_CSV) == 0:
            w.writeheader()
        w.writerow(row)
    print(f"appended to {RESULTS_CSV}")


if __name__ == "__main__":
    main()
