"""物理可预测性探针：本数据集里 POR/PERM/SW 是否真受石油物理约束支配？

目的（研究用，非交付）：决定"物理信息"这条路是否值得投。关键判据：
  1. 真实段内 POR~PERM 的 Archie/Kozeny 关系强度（Spearman 单调相关）；
  2. Wyllie 时均方程能否从时差曲线解释孔隙度方差（R^2）；
  3. ARCHIE 从 RT/DEN/POR 推 SW 与真值的 log10 误差 vs 当前 PERM 口径容差；
  4. 三个目标之间的**可预测方差比例**：给定其他两个目标，还剩多少不可约方差？

只做描述统计，不 fit、不 import torch。
"""
from __future__ import annotations
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.feature import load_wells, make_targets, default_mask
from src.utils import EPS, TARGETS

TOL = {"POR": 0.08, "PERM": 1.0, "SW": 0.05}
_wells = load_wells(os.path.join(ROOT, "train"), has_labels=True)
wells = pd.concat(_wells, ignore_index=True)
targets = make_targets(wells)

cols = [c for c in wells.columns if c not in ("logId", "DEPTH")]
print("曲线列:", cols)
print("井数:", wells["logId"].nunique(), " 行数:", len(wells))

_m = default_mask(targets)
real = ~(_m["POR"] & _m["PERM"] & _m["SW"])
w = wells.loc[real]
t = targets.loc[real]
print(f"\n真实段: {real.mean():.3f} 占比, {real.sum():,} 行")

# ---- 1. 目标间单调相关（Spearman）----
print("\n=== 真实段内 Spearman 相关（目标 x 目标）===")
print(t[["POR", "PERM", "SW"]].corr(method="spearman").round(3).to_string())

print("\n=== 目标 ~ 曲线 Spearman |r| Top5 ===")
for tgt in ["POR", "PERM", "SW"]:
    cs = w[cols].corr(method="spearman")[tgt].dropna().abs().sort_values(ascending=False)
    print(f"  {tgt}: " + ", ".join(f"{k}={v:.3f}" for k, v in cs.head(5).items()))

# ---- 2. Wyllie 时均方程  phi_w = (Dt - Dm) / (Df - Dm) ----
print("\n=== Wyllie 时均方程对 POR 的解释力（真实段，按井）===")
need = [c for c in ("DT", "DEN", "POR") if c in w.columns]
if len(need) == 3:
    Dm = 0.040  # 砂岩骨架 ~55-65 us/ft... 用 s/m 单位需换算，这里只看解释力上限
    Df = 0.090  # 水
    d = w[["DT", "DEN", "POR"]].dropna()
    d = d[(d["DT"] > 0) & (d["POR"] > 0)]
    phi_wyllie = np.clip((d["DT"] - Dm) / (Df - Dm), 0, 1) * 100  # 百分数
    r = float(np.corrcoef(phi_wyllie, d["POR"])[0, 1])
    print(f"  Dm={Dm} Df={Df} us/ft 单位假设: corr(phi_wyllie, POR) = {r:.3f}")
    print(f"  POR 真值 std={d['POR'].std():.3f}, Wyllie 残差 std={np.std(d['POR'].values - phi_wyllie):.3f}")
else:
    print("  缺 DT/DEN 列，跳过")

# ---- 3. Archie 从 RT/DEN/POR 推 SW ----
print("\n=== Archie SW 推算 vs 真值（真实段，RT/DEN/POR 齐全行）===")
if {"RT", "DEN", "POR", "SW"}.issubset(w.columns):
    d = w[["RT", "DEN", "POR", "SW"]].dropna()
    d = d[(d["RT"] > 0) & (d["POR"] > 0) & (d["SW"] > 0)]
    Rw = 0.05  # 地层水电阻率（假设）
    Sw_archie = np.clip(100 * (Rw / (d["RT"].values * d["POR"].values ** 2)) ** 0.5, 0, 100)
    err = np.abs(Sw_archie - d["SW"].values) / (TOL["SW"] * (np.abs(d["SW"].values) + EPS))
    sw_acc = float(np.maximum(0, 1 - err).mean())
    print(f"  行数={len(d):,}  n=m=2, Rw=0.05 假设")
    print(f"  median |Sw_archie - Sw_true|/tol = {np.median(err):.2f}  -> Acc_SW 若直接用 Archie = {sw_acc:.3f}")
    print(f"  median 比值 Sw_archie/Sw_true = {np.median(Sw_archie / d['SW'].values):.3f}")
else:
    print("  缺 RT/DEN 列，跳过")

# ---- 4. 互信息式分解：给定其他两目标还剩多少不可约方差 ----
print("\n=== 三个目标的耦合度：给定其他两目标后，本目标残差相对降幅 ===")
sub = t[["POR", "PERM", "SW"]].dropna()
sub = sub[(sub["POR"] > 0) & (sub["PERM"] > 0) & (sub["SW"] > 0)]
sub_p = np.log10(sub["PERM"].values)  # log10 空间，PERM 跨量级
base = {"POR": sub["POR"].values, "PERM": sub_p, "SW": sub["SW"].values}

def resid_var(y, preds):
    """线性投影后的残差方差 / 原始方差（= 1 - R^2）。"""
    A = np.column_stack([np.ones(len(y))] + [p for p in preds])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    r = y - A @ coef
    return float(r.var() / y.var())

for tgt in ["POR", "PERM", "SW"]:
    y = base[tgt]
    oth = [base[o] for o in ["POR", "PERM", "SW"] if o != tgt]
    r2_both = 1 - resid_var(y, oth)
    r2_first = 1 - resid_var(y, [oth[0]])
    print(f"  {tgt:5s}: R^2(全部其他目标) = {r2_both:.3f}   R^2(仅首个) = {r2_first:.3f}"
          f"   -> 不可约残差占比 = {1 - r2_both:.3f}")

# ---- 5. 当前 PERM 口径下，POR 预测精度如何传导到 PERM ----
print("\n=== PERM 对 POR 的局部弹性（真实段，log-log 局部斜率）===")
if len(sub) > 0:
    p, l, s = sub["POR"].values, sub_p, sub["SW"].values
    for tgt_name, tgt_val in [("POR", p), ("SW", s)]:
        mask = np.abs(np.diff(np.log10(tgt_val + EPS))) > 1e-9
        dlog = np.diff(l)[mask]
        dsrc = np.diff(np.log10(tgt_val + EPS))[mask]
        good = np.isfinite(dlog) & np.isfinite(dsrc) & (np.abs(dsrc) > 0.05)
        if good.sum() > 100:
            slope = float(np.sum(dlog[good] * dsrc[good]) / np.sum(dsrc[good] ** 2))
            print(f"  dlog10(PERM)/dlog10({tgt_name}) = {slope:.2f}  (n={good.sum():,})")
