"""边界噪声诊断：real 点里有多少个在官方容差带上无法与默认值区分。

AUC 上界推导：若 real 点中比例 f 的曲线特征与默认段点不可区分（标签为
isclose 边界内的硬币翻转），则
    AUC_max ≈ (1-f)*1 + f*0.5 = 1 - 0.5f
任何模型（无论感受野多长）都超过不了它。
"""
import os, sys, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from src.feature import load_wells, make_targets, default_mask
from src.utils import TARGETS, DEFAULT_VALUES, EPS

wells = load_wells("./train", has_labels=True)
ids = [w["logId"].iloc[0] for w in wells]
tr = pd.concat([w for w in wells if w["logId"].iloc[0] in set(ids[:72])], ignore_index=True)
va = pd.concat([w for w in wells if w["logId"].iloc[0] in set(ids[72:])], ignore_index=True)

TOL = {"POR": 0.08, "SW": 0.05}
print(f"{'target':6s}{'n_real':>9s}{'f<0.5%':>9s}{'f<1%':>8s}{'f<2%':>8s}{'f<5%':>8s}  AUC_max(<2%)")
for c in TARGETS:
    for tag, df in (("train", tr), ("val", va)):
        Y = make_targets(df); D = default_mask(Y)
        y = Y[c].to_numpy(float); d = D[c].to_numpy(bool)
        real = (~np.isnan(y)) & (~d)
        dv = DEFAULT_VALUES[c]
        if c == "PERM":
            dist = np.abs(np.log10(np.maximum(y[real], EPS) / dv))   # decades
            bands = [0.005, 0.01, 0.02, 0.05]
        else:
            dist = np.abs(y[real] - dv) / (dv + EPS)                  # relative
            bands = [0.005, 0.01, 0.02, 0.05]
        n = int(real.sum())
        fs = [float((dist < b).mean()) for b in bands]
        print(f"{c+'/'+tag:12s}{n:9d}{fs[0]:9.3f}{fs[1]:8.3f}{fs[2]:8.3f}{fs[3]:8.3f}  {1-0.5*fs[2]:.4f}")

# 直接看真实段里 POR 的取值分布
print("\nPOR 真实段取值分布（train）:")
Y = make_targets(tr); D = default_mask(Y)
y = Y["POR"].to_numpy(float); real = (~np.isnan(y)) & (~D["POR"].to_numpy(bool))
q = np.quantile(y[real], [0, .01, .05, .1, .25, .5, .75, .9, .95, .99, 1])
print("POR real quantiles:", np.round(q, 5))
print("  距默认值 0.1 的相对距离分位:", np.round(np.abs(q-0.1)/0.1, 5))
print("  距默认 <0.02 的 real 点比例:", float((np.abs(y[real]-0.1)/0.1 < 0.02).mean()))
print("\nPERM 真实段 log10 分布（train）:")
Y2 = make_targets(tr); D2 = default_mask(Y2)
y2 = Y2["PERM"].to_numpy(float); r2 = (~np.isnan(y2)) & (~D2["PERM"].to_numpy(bool))
print("log10 PERM real quantiles:", np.round(np.log10(np.maximum(y2[r2],EPS)),3))
