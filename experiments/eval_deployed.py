"""直接评估已部署 models/model.pkl 在 screen 验证井上的真实分数（端到端 predict()）.

目的：消除"scren 调优工件(buggy LN10) vs 部署工件(train.py 正确 LN10)"的同源疑问，
给出交付模型的实际 screen 分数。
"""
from __future__ import annotations
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.feature import load_wells, make_features, make_targets, default_mask
from src.model import load_model, predict
from src.utils import DEFAULT_VALUES, EPS, TARGETS, TOLERANCE, WEIGHTS


def pw(c, pred, true):
    p = np.asarray(pred, float); t = np.asarray(true, float)
    if c == "PERM":
        p = np.log10(np.maximum(p, EPS)); t = np.log10(np.maximum(t, EPS))
    return np.maximum(0.0, 1.0 - np.abs(p - t) / (TOLERANCE[c] * (np.abs(t) + EPS)))


def main():
    model = load_model("models/model.pkl")
    print("deployed: blend_w=", model.get("blend_w"), "clf_tcn_w=", model.get("clf_tcn_w"),
          "thresholds=", model.get("thresholds"), "n_tcn=", len(model.get("tcn_sds") or []))
    wells = load_wells("./train", has_labels=True); ids = [w["logId"].iloc[0] for w in wells]
    va = [w for w in wells if w["logId"].iloc[0] in set(ids[72:])]
    df = pd.concat(va, ignore_index=True)
    X = make_features(df, context=True, priors=True)
    Y = make_targets(df); td = default_mask(Y)
    out = predict(model, X)
    tot, parts = 0.0, {}
    for c in TARGETS:
        y = Y[c].to_numpy(); m = ~np.isnan(y)
        p = out[c].to_numpy()
        acc = pw(c, p[m], y[m]).mean()
        d = pw(c, p[m & td[c].to_numpy()], y[m & td[c].to_numpy()]).mean()
        r = pw(c, p[m & ~td[c].to_numpy()], y[m & ~td[c].to_numpy()]).mean()
        parts[c] = (acc, r, d); tot += WEIGHTS[c] * acc
    print(f"DEPLOYED screen total={tot*100:.3f}")
    print("  " + " | ".join(f"{c}: acc={parts[c][0]:.4f} real={parts[c][1]:.4f} def={parts[c][2]:.4f}" for c in TARGETS))
    v3 = 68.545
    print(f"  vs v3(68.545): {tot*100 - v3:+.3f}")


if __name__ == "__main__":
    main()
