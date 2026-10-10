"""残差二阶段可修正性：一阶段残差里到底还有多少可预测信号？

动机
----
官方指标是 max(0, 1 − |ŷ−y|/(tol·|y|))，**带外梯度恒为 0**。一阶段训练目标里凡是不在
容差带内的点都得不到任何学习信号 —— 而 POR/SW 真实段 acc 只有 0.22/0.19，绝大多数点早已
在带外，因此"最差的错"恰恰是永远不会被修正的那部分。这是训练目标的病态，不是信息上限。

同时已知一阶段在真实段上有系统性乘性偏差（pred/true 中位 POR 1.069、PERM 0.671、SW 1.023）。
纯全局乘法因子**在数学上无解**：若 ŷ = m·y，则 |ŷ−y|/(tol·|y|) = |m−1|/tol 与 y 无关，
全点同值，最优 m 恰为 1.0。所以偏差必须是**依赖特征**的，唯一机制是残差二阶段。

本脚本做生死判据（全部在**真实段**上，用 5 折 OOF 残差，与部署同构）：
  S1 残差 vs 自身滞后        —— 残差自相关高说明时序上可预测
  S2 残差 vs 一阶段特征      —— 单特征 AUC/相关；若全 ≈ 0，同特征二阶段必然无信号
  S3 逐井残差中位           —— 若井间方差大，残差主要是"井级偏差"而非点级噪声
  S4 带内/带外分解          —— 带外点残差是否更极端（带外梯度为 0 的直接后果）
  S5 二阶段实测             —— HistGB 预测残差，按最优线性融合，看真实 acc 能否移动
  S6 MSE 一阶段 vs 截断目标 —— 部署的一阶段已用 metric_loss；对比 Huber 与 trunc-hinge

用法: python experiments/diag_second_stage.py
"""
from __future__ import annotations

import os
import sys
import gc
import time
import warnings
import resource

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold

from src.feature import load_wells, make_targets, default_mask, make_features
from src.utils import TARGETS, WEIGHTS, EPS

CURVE_COLS = ["GR", "DEN", "CNL", "AC", "NPHI", "DT", "SP", "RT", "RT2", "ILD10", "ILD30", "CALIP", "BIT"]
RNG = np.random.default_rng(42)
TOL = {"POR": 0.08, "PERM": 1.0, "SW": 0.05}


def rss() -> str:
    # 本机 macOS 的 ru_maxrss 单位是字节
    return f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576:.0f}MB"


def acc_vec(c: str, p: np.ndarray, t: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    t = np.asarray(t, float)
    if c == "PERM":
        return np.maximum(0.0, 1.0 - np.abs(np.log10(np.maximum(p / np.maximum(t, EPS), EPS))))
    tol = TOL[c]
    return np.maximum(0.0, 1.0 - np.abs(p - t) / (tol * (np.abs(t) + EPS)))


def logify(c: str, arr: np.ndarray) -> np.ndarray:
    return np.log10(np.clip(np.asarray(arr, float), 1e-9, None))


def build_feats(df: pd.DataFrame) -> np.ndarray:
    # make_features(context=True, priors=True) 已含 237 列（225 上下文 + 12 wellz 末尾）
    return make_features(df, context=True, priors=True).to_numpy(dtype=np.float32)


def hgb(loss: str):
    return HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.06, max_depth=None, max_leaf_nodes=63,
        l2_regularization=1.0, min_samples_leaf=40, loss=loss, random_state=42)


def trunc_hinge(y_true, y_pred, tol):
    """带内线性、带外 clamp 到 0 —— 官方逐点指标的逐点形式（对数空间，PERM 原生）。"""
    d = np.abs(y_true - y_pred)
    return np.maximum(0.0, 1.0 - d / tol)


def main() -> None:
    t0 = time.time()
    wells = load_wells("./train", has_labels=True)
    ids = [w["logId"].iloc[0] for w in wells]
    tr = pd.concat([w for w in wells if w["logId"].iloc[0] in set(ids[:72])], ignore_index=True)
    va = pd.concat([w for w in wells if w["logId"].iloc[0] in set(ids[72:])], ignore_index=True)
    del wells
    gc.collect()
    tr = tr.sort_index().reset_index(drop=True)
    va = va.sort_index().reset_index(drop=True)
    ytr = make_targets(tr)
    yva = make_targets(va)
    dtr = default_mask(ytr)
    dva = default_mask(yva)
    print(f"训练 {len(tr):,} 点 / 验证 {len(va):,} 点   {rss()}")

    Xtr = build_feats(tr)
    Xva = build_feats(va)
    print(f"特征 {Xtr.shape[1]}（含 12 列 wellz）   {rss()}")

    # 训练井的井级标识：用于评估残差的井间方差
    wtr = tr["logId"].to_numpy()
    wva = va["logId"].to_numpy()

    print("\n" + "=" * 92)
    print("一阶段（5 折 OOF，真实段，log10 空间）+ 二阶段残差可预测性")
    print("=" * 92)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_ids = np.zeros(len(tr), dtype=int)
    for fi, (_, oi) in enumerate(kf.split(Xtr)):
        fold_ids[oi] = fi

    for c in TARGETS:
        tol = TOL[c]
        yr = logify(c, ytr[c].to_numpy(float))
        yv = logify(c, yva[c].to_numpy(float))
        # 真实段（非默认段）。部署的回归器只在真实段上 fit，这里保持一致。
        real_tr = np.isfinite(yr) & np.isfinite(Xtr).all(axis=1) & ~dtr[c].to_numpy()
        real_va = np.isfinite(yv) & np.isfinite(Xva).all(axis=1) & ~dva[c].to_numpy()
        print(f"\n{'─' * 92}\n目标 {c}   真实段 训练 {real_tr.sum():,} / 验证 {real_va.sum():,}")

        # ---- 一阶段：全量训练集 fit，验证集前向 ----
        m_full = hgb("squared_error")
        m_full.fit(Xtr[real_tr], yr[real_tr])
        pv = m_full.predict(Xva[real_va])
        tv = yv[real_va]
        base = acc_vec(c, np.power(10.0, pv), np.power(10.0, tv))

        # ---- 一阶段 OOF（真实段，用于残差诊断）----
        p_oof = np.full(len(tr), np.nan)
        for fi in range(5):
            oi = fold_ids == fi
            ii = ~oi
            m = hgb("squared_error")
            m.fit(Xtr[real_tr & ii], yr[real_tr & ii])
            p_oof[oi] = m.predict(Xtr[oi])
            del m
            gc.collect()
        ro = p_oof[real_tr] - yr[real_tr]
        xo = Xtr[real_tr]
        po = p_oof[real_tr]
        print(f"  一阶段真实段 OOF  log10-MAE {np.nanmean(np.abs(ro)):.4f}  "
              f"相对误差中位 {np.nanmedian(np.abs(np.power(10.0, ro) - 1)) * 100:.2f}%"
              f"   | 验证集 acc = {base.mean():.4f}")

        # S1 残差自相关（一阶段特征之外还剩多少时序信号）
        ac = [float(np.corrcoef(ro[:-k], ro[k:])[0, 1]) for k in (1, 5, 20)]
        print(f"  S1 残差自相关  lag1 {ac[0]:+.3f}  lag5 {ac[1]:+.3f}  lag20 {ac[2]:+.3f}"
              f"   → 残差沿深度是否仍系统性")

        # S2 残差 vs 一阶段特征（同特征二阶段的信号上限）
        best_c, best_f = 0.0, -1
        for j in range(xo.shape[1]):
            x = xo[:, j]
            m2 = np.isfinite(x) & np.isfinite(ro)
            if m2.sum() < 500 or np.ptp(x[m2]) < 1e-12:
                continue
            cv = float(np.corrcoef(x[m2], ro[m2])[0, 1])
            if abs(cv) > abs(best_c):
                best_c, best_f = cv, j
        print(f"  S2 残差 vs 一阶段特征  最大 |Pearson| {abs(best_c):.4f}（列 {best_f}）"
              f"   → 同特征二阶段的可预测上限")

        # S3 逐井残差（井级乘性偏差 vs 点级噪声）
        wl = wtr[real_tr]
        uw = np.unique(wl)
        med = np.array([float(np.nanmedian(ro[wl == w])) for w in uw])
        within = float(np.sqrt(np.nanmean([np.nanvar(ro[wl == w]) for w in uw])))
        print(f"  S3 逐井残差中位：跨井 std {med.std():.4f} / 井内 std {within:.4f}"
              f" / 跨井极差 {med.max() - med.min():+.4f}"
              f"   → 跨井 std 大说明主要是井级乘性偏差")

        # S4 带内/带外分解（带外梯度为 0 的直接后果）
        inband = np.abs(ro) <= tol
        out = np.abs(ro) > tol
        print(f"  S4 带内占 {inband.mean() * 100:5.1f}%   带外 |残差| 中位 {np.median(np.abs(ro[out])):.3f}"
              f" / 容差 {tol}   带内 |残差| 中位 {np.median(np.abs(ro[inband])):.3f}")
        bins = np.quantile(po, np.linspace(0, 1, 6))
        row = "     按一阶段预测 log10 分箱的带外率："
        for b in range(5):
            m5 = (po >= bins[b]) & (po < bins[b + 1])
            if m5.sum() > 100:
                row += f" [{bins[b]:+.2f},{bins[b + 1]:+.2f}) {out[m5].mean() * 100:4.0f}%"
        print(row)

        # S5 二阶段实测：HistGB 预测残差，按 λ 线性融合回一阶段
        m2f = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05,
                                            max_leaf_nodes=31, min_samples_leaf=80,
                                            l2_regularization=1.0, loss="squared_error",
                                            random_state=42)
        m2f.fit(xo, ro)
        r2 = m2f.predict(Xva[real_va])
        print(f"  S5 二阶段残差预测 log10-MAE {np.nanmean(np.abs(r2)):.4f}"
              f"   （一阶段验证集 {np.nanmean(np.abs(pv - tv)):.4f}）")
        for lam in (0.25, 0.5, 0.75, 1.0, 1.5):
            a = acc_vec(c, np.power(10.0, pv - lam * r2), np.power(10.0, tv))
            print(f"      λ={lam:<4} 真实段 acc = {a.mean():.4f}"
                  f"   Δ = {(a.mean() - base.mean()) * 100 * WEIGHTS[c]:+.4f} 分")

        # S6 换一阶段训练目标：Huber / trunc-hinge（= 官方逐点指标）
        for loss in ("squared_error", "huber"):
            m = hgb(loss)
            m.fit(Xtr[real_tr], yr[real_tr])
            a = acc_vec(c, np.power(10.0, m.predict(Xva[real_va])), np.power(10.0, tv))
            print(f"      一阶段 loss={loss:<14} 真实段 acc = {a.mean():.4f}"
                  f"  Δ = {(a.mean() - base.mean()) * 100 * WEIGHTS[c]:+.4f} 分")
        m = hgb("squared_error")
        yt2 = trunc_hinge(yr[real_tr], yr[real_tr], tol)
        m.fit(Xtr[real_tr], yt2)
        a = acc_vec(c, np.power(10.0, m.predict(Xva[real_va])), np.power(10.0, tv))
        print(f"      一阶段 loss=trunc_hinge 真实段 acc = {a.mean():.4f}"
              f"  Δ = {(a.mean() - base.mean()) * 100 * WEIGHTS[c]:+.4f} 分")
        del m_full, m2f
        gc.collect()

    print(f"\n用时 {time.time() - t0:.0f}s   峰值 {rss()}")


if __name__ == "__main__":
    main()
