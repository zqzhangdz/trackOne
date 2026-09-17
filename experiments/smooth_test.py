"""screen 切分：对融合回归值(log10空间)按井做窗口平滑的增益测试."""
from __future__ import annotations
import os, sys, warnings
warnings.filterwarnings("ignore")
import joblib, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.feature import default_mask, load_wells, make_features, make_targets
from src.model import TwoStageModel
from src.utils import DEFAULT_VALUES, EPS, TARGETS, TOLERANCE, WEIGHTS
from solution.tcn_model import numpy_forward

def pw(c, p, t):
    p = np.asarray(p, float); t = np.asarray(t, float)
    if c == "PERM":
        p = np.log10(np.maximum(p, EPS)); t = np.log10(np.maximum(t, EPS))
    return np.maximum(0.0, 1.0 - np.abs(p - t) / (TOLERANCE[c] * (np.abs(t) + EPS)))

def main(w_tcn=0.5, tcn_tag="T3_sel80"):
    wells = load_wells("./train", has_labels=True)
    ids = [w["logId"].iloc[0] for w in wells]
    tr = [w for w in wells if w["logId"].iloc[0] in set(ids[:72])]
    va = [w for w in wells if w["logId"].iloc[0] in set(ids[72:])]
    tr_df, va_df = pd.concat(tr, ignore_index=True), pd.concat(va, ignore_index=True)
    Xtr = make_features(tr_df, context=True, priors=True)
    Xva = make_features(va_df, context=True, priors=True)
    Ytr, Yva = make_targets(tr_df), make_targets(va_df)
    model = TwoStageModel(monotonic=True).fit(Xtr, Ytr, default_mask(Ytr))
    from src.feature import inverse_transform_target
    reg_tree_log = {c: np.log10(np.maximum(inverse_transform_target(
        c, pd.Series(model.regressors[c].predict(Xva), index=Xva.index)).to_numpy(), EPS)) for c in TARGETS}
    art = joblib.load(f"experiments/runs/tcn_{tcn_tag}.joblib")
    medians, stds = art["medians"], art["stds"]
    reg_tcn_log = {c: np.zeros(len(va_df)) for c in TARGETS}
    va_ids = va_df["logId"].to_numpy()
    for w in va:
        Xw = make_features(w, context=False, priors=True).fillna(medians)
        Xw = ((Xw - stds.replace(0, 1.0)) / stds.replace(0, 1.0)).to_numpy(dtype=np.float32)
        o = numpy_forward(Xw, art["sd"], art["dilations"])
        m = va_ids == w["logId"].iloc[0]
        for c in TARGETS:
            reg_tcn_log[c][m] = o[c][:, 1]
    blend = {c: (1 - w_tcn) * reg_tree_log[c] + w_tcn * reg_tcn_log[c] for c in TARGETS}
    p_def = {c: model.classifiers[c].predict_proba(Xva)[:, 1] for c in TARGETS}

    def smooth_per_well(arr, win):
        if win <= 1:
            return arr
        out = arr.copy()
        for lid in pd.unique(va_ids):
            m = va_ids == lid
            s = pd.Series(arr[m])
            out[m] = s.rolling(win, center=True, min_periods=1).mean().to_numpy()
        return out

    grid = np.arange(0.05, 0.96, 0.05)
    for win in (1, 3, 5, 9, 15, 25):
        regs = {c: smooth_per_well(blend[c], win) for c in TARGETS}
        tot = 0.0
        parts = []
        for c in TARGETS:
            y = Yva[c].to_numpy(); m = ~np.isnan(y)
            best_t, best_a = 0.5, -1.0
            for t in grid:
                pred = np.where(p_def[c] >= t, DEFAULT_VALUES[c], np.power(10.0, regs[c]))
                a = pw(c, pred[m], y[m]).mean()
                if a > best_a: best_a, best_t = a, float(t)
            tot += WEIGHTS[c] * best_a
            parts.append(f"{c}={best_a:.4f}@{best_t:.2f}")
        print(f"win={win:2d}: total={tot * 100:.3f} | " + " ".join(parts))

if __name__ == "__main__":
    main()
