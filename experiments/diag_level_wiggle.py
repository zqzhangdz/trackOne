"""TCN/树 误差分解：真实段内 level 误差 vs wiggle 误差（screen 验证井）.

对每个“真实连续段”(truth 分段) 计算:
  level_err  = mean(pred) - mean(true)        (log10 空间)
  wiggle_err = (pred - mean(pred)) - (true - mean(true))
指标容差 = 0.08*|log10 y| (PERM) / 0.08 (POR, log10 空间) / 0.05*|y| (SW, 线性)
输出各误差分量的 RMS 与对最终 acc 的相对影响。
"""
from __future__ import annotations
import os, sys, warnings
warnings.filterwarnings("ignore")
import joblib, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.feature import default_mask, load_wells, make_features, make_targets
from src.utils import DEFAULT_VALUES, EPS, TARGETS, TOLERANCE, WEIGHTS
from solution.tcn_model import numpy_forward
RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")

def main():
    wells = load_wells("./train", has_labels=True)
    ids = [w["logId"].iloc[0] for w in wells]
    va = [w for w in wells if w["logId"].iloc[0] in set(ids[72:])]
    va_df = pd.concat(va, ignore_index=True); Yva = make_targets(va_df)
    tdef = {c: default_mask(Yva)[c].to_numpy() for c in TARGETS}
    larr = va_df["logId"].to_numpy()
    cache = joblib.load(os.path.join(RUNS, "screen_tree_cache.joblib"))
    reg_tree = cache["reg"]
    art = joblib.load(os.path.join(RUNS, "tcn_T6_long.joblib"))
    reg_tcn = {}
    for w in va:
        Xw = make_features(w, context=False, priors=True).fillna(art["medians"])
        Xw = ((Xw - art["stds"].replace(0, 1.0)) / art["stds"].replace(0, 1.0)).to_numpy(np.float32)
        o = numpy_forward(Xw, art["sd"], art["dilations"])
        for c in TARGETS:
            reg_tcn.setdefault(c, []).append(o[c][:, 1])
    reg_tcn = {c: np.concatenate(v) for c, v in reg_tcn.items()}

    for name, regmap in [("tree", reg_tree), ("tcn", reg_tcn),
                         ("mix.35", {c: 0.3*reg_tree[c]+0.7*reg_tcn[c] for c in TARGETS})]:
        print(f"=== {name} ===")
        for c in TARGETS:
            y = Yva[c].to_numpy()
            yl = np.log10(np.maximum(y, EPS)) if c == "PERM" else y
            if c != "PERM":
                yl = np.log10(np.maximum(y, EPS))  # 统一 log10
            pred = regmap[c]
            lev, wig = [], []
            for lid in pd.unique(larr):
                m = (larr == lid) & ~tdef[c] & ~np.isnan(y)
                idx = np.flatnonzero(m)
                if idx.size == 0: continue
                brk = np.flatnonzero(np.diff(idx) != 1)
                for g in np.split(idx, brk + 1):
                    if len(g) < 3: continue
                    pt = pred[g]; tt = yl[g]
                    lev.append(pt.mean() - tt.mean())
                    wig.extend((pt - pt.mean()) - (tt - tt.mean()))
            lev = np.array(lev); wig = np.array(wig)
            # 容差尺度（log10 空间）
            tol = {"POR": 0.08, "PERM": 0.08, "SW": 0.05}[c]
            print(f"  {c}: nseg={len(lev)} levelRMS={np.sqrt((lev**2).mean()):.4f} wiggleRMS={np.sqrt((wig**2).mean()):.4f} "
                  f"| levelRMS/tol={np.sqrt((lev**2).mean())/tol:.2f} wiggleRMS/tol={np.sqrt((wig**2).mean())/tol:.2f}")

if __name__ == "__main__":
    main()
