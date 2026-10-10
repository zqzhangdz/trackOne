"""跨目标信息可用性测试：POR~PERM 强耦合（R^2=0.86）能否转化为**分数**提升？

背景（research_physics_probe.py 的结论）
---------------------------------------
真实段内 POR 与 PERM 的 Spearman = 0.73，R^2(POR | PERM) = R^2(PERM | POR) = 0.863。
但当前管线里三个目标**各自独立预测**：POR 回归器看不到 PERM 的预测。
而 PERM 的 real acc (0.513) 远好于 POR (0.17~0.22) —— PERM 预测里可能装着
POR 自身曲线给不出的信息。

本脚本测**残差层面**的可用量（分数层面的东西，不是原始方差）：
  A: POR_pred ~ logs                 （当前做法，72 井训练 / 8 井 screen）
  B: POR_pred ~ logs + PERM_pred     （加上 PERM 预测这一列 = 可达）
  C: POR_pred ~ logs + PERM_true     （不可达上界：用真值 PERM）
三者都在**官方口径真实段**上算 acc。B-A 是可达部分，C-A 是理论上限。

SW 作为负对照：SW 的 R^2(POR,PERM) 只有 0.088，预期 B≈A。

只 fit HistGB，不 import torch（§5.1）。
"""
from __future__ import annotations
import os, sys, gc, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sklearn.ensemble import HistGradientBoostingRegressor as HGB
from src.feature import load_wells, make_targets, default_mask
from src.utils import EPS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOL = {"POR": 0.08, "PERM": 1.0, "SW": 0.05}
CURVES = ["GR", "PE", "SP", "CAL", "AC", "DEN", "CNL", "RXO", "RT", "DEVI", "AZIM", "BIT"]


def feat(df: pd.DataFrame, add: dict | None = None) -> pd.DataFrame:
    """点式特征（无上下文，快）：DEPTH + 12 曲线 + 7 个物理变换列。"""
    X = df[["DEPTH"] + CURVES].copy()
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X["logRT"] = np.log10(X["RT"].clip(lower=1e-8))
    X["logRXO"] = np.log10(X["RXO"].clip(lower=1e-8))
    X["RT_over_RXO"] = X["RT"] / X["RXO"].replace(0, np.nan)
    X["POR_D"] = (2.65 - X["DEN"]) / 1.65 * 100.0
    X["POR_S"] = (X["AC"] - 55.5) / (189.0 - 55.5) * 100.0
    X["POR_avg"] = (X["POR_D"] + X["CNL"]) / 2.0
    X["POR_ds"] = X["POR_D"] - X["POR_S"]
    if add:
        for k, v in add.items():
            X[k] = np.asarray(v, float)
    return X


def acc_target(tgt: str, pred, true) -> float:
    p, t = np.asarray(pred, float), np.asarray(true, float)
    ok = np.isfinite(p) & np.isfinite(t)
    p, t = p[ok], t[ok]
    if p.size == 0:
        return float("nan")
    if tgt == "PERM":
        a = np.maximum(0.0, 1.0 - np.abs(np.log10(np.maximum(p / np.maximum(t, EPS), EPS))))
    else:
        a = np.maximum(0.0, 1.0 - np.abs(p - t) / (TOL[tgt] * (np.abs(t) + EPS)))
    return float(a.mean())


def hgb(**kw):
    return HGB(max_iter=400, learning_rate=0.05, max_leaf_nodes=63,
               l2_regularization=1.0, random_state=0, **kw)


def main():
    t0 = time.time()
    ws = load_wells(os.path.join(ROOT, "train"), has_labels=True)
    ids = [w["logId"].iloc[0] for w in ws]
    cut = set(ids[:72])
    tr = pd.concat([w for w, i in zip(ws, ids) if i in cut], ignore_index=True)
    va = pd.concat([w for w, i in zip(ws, ids) if i not in cut], ignore_index=True)
    YT, YV = make_targets(tr), make_targets(va)
    rm = ~((default_mask(YT)["POR"]) & (default_mask(YT)["PERM"]) & (default_mask(YT)["SW"]))
    vm = ~((default_mask(YV)["POR"]) & (default_mask(YV)["PERM"]) & (default_mask(YV)["SW"]))
    print(f"72/8 切分: train 真实段 {rm.sum():,}, val 真实段 {vm.sum():,}  ({time.time()-t0:.0f}s)")

    Xtr = feat(tr)
    Xva = feat(va)
    Xtr_r, Xva_r = Xtr.loc[rm], Xva.loc[vm]

    # ---- 先建 PERM 预测器（当前管线里 PERM 最好预测：real acc ~0.51）----
    # 注意：预测结果必须对齐到 DataFrame 原始 index，后面要作为列拼回去。
    rmp = rm & YT["PERM"].notna()
    perm_log = np.log10(YT.loc[rmp, "PERM"].clip(lower=EPS).to_numpy())
    pm = hgb()
    pm.fit(Xtr.loc[rmp], perm_log)
    perm_pred_tr = pd.Series(np.clip(10 ** pm.predict(Xtr), EPS, None), index=Xtr.index)
    perm_pred_val = pd.Series(np.clip(10 ** pm.predict(Xva), EPS, None), index=Xva.index)
    perm_true_val = YV.loc[vm, "PERM"].to_numpy()
    print(f"  PERM real acc (72→8) = {acc_target('PERM', perm_pred_val[vm].to_numpy(), perm_true_val):.4f}")

    # ---- 物理耦合探针：给定 PERM，POR 还剩多少不可约残差？ ----
    print("\n=== 给定 PERM 预测后，POR 目标本身的线性可解释比例 ===")
    por_true_val = YV.loc[vm, "POR"].to_numpy()
    for nm, src in [("PERM_true", perm_true_val), ("PERM_pred", perm_pred_val[vm].to_numpy())]:
        y = por_true_val
        s = np.asarray(src, float)
        ok = np.isfinite(y) & np.isfinite(s)
        y, s = y[ok], s[ok]
        base_var = np.var(y)
        A = np.column_stack([np.ones(len(s)), s, np.log10(np.maximum(s, EPS))])
        c, *_ = np.linalg.lstsq(A, y, rcond=None)
        r2 = 1 - np.var(y - A @ c) / base_var
        print(f"  POR ~ {nm}: R^2 = {r2:.3f} (n={len(y):,})")

    W = {"POR": 0.30, "SW": 0.35}
    for tgt in ["POR", "SW"]:
        rt = rm & YT[tgt].notna() & YT["PERM"].notna()
        yt_tr = YT.loc[rt, tgt].to_numpy()
        yt_va = YV.loc[vm, tgt].to_numpy()
        Xb_tr = Xtr.copy(); Xb_tr["PERM_pred"] = perm_pred_tr
        Xb_va = Xva.copy(); Xb_va["PERM_pred"] = perm_pred_val
        perm_true_series = pd.Series(np.nan, index=Xtr.index)
        perm_true_series.loc[rm] = YT.loc[rm, "PERM"].to_numpy()
        Xc_tr = Xtr.copy(); Xc_tr["PERM_true"] = perm_true_series
        perm_true_va = pd.Series(np.nan, index=Xva.index)
        perm_true_va.loc[vm] = perm_true_val
        Xc_va = Xva.copy(); Xc_va["PERM_true"] = perm_true_va
        rows = [
            ("A: logs only",        Xtr.loc[rt], Xva.loc[vm]),
            ("B: logs + PERM_pred", Xb_tr.loc[rt], Xb_va.loc[vm]),
            ("C: logs + PERM_true", Xc_tr.loc[rt], Xc_va.loc[vm]),
        ]
        out = {}
        print(f"\n--- {tgt} (权重 {W[tgt]}, train 行数 {len(yt_tr):,}) ---")
        for nm, xtr, xva in rows:
            g = hgb()
            g.fit(xtr, yt_tr)
            a = acc_target(tgt, g.predict(xva), yt_va)
            out[nm] = a
            print(f"  {nm}: real acc = {a:.4f}")
            gc.collect()
        d_ab = out["B: logs + PERM_pred"] - out["A: logs only"]
        d_ac = out["C: logs + PERM_true"] - out["A: logs only"]
        frac = d_ab / d_ac if d_ac > 1e-9 else float("nan")
        print(f"  >> B-A = {d_ab:+.4f}  |  C-A = {d_ac:+.4f}  |  可达比例 B/C = {frac:.2f}")
        print(f"  >> 折算总分: B-A × {W[tgt]} = {d_ab*W[tgt]*100:+.3f} 分 "
              f"(screen 口径, 仅真实段)")
    print(f"\n总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
