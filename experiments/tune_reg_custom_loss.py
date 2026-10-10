"""把官方指标本身作为训练目标：自定义 BaseLoss 直接优化 Acc。

这是本项目从未碰过的轴
----------------------
之前所有回归器实验都在 {squared_error, absolute_error, gamma, poisson, quantile}
里挑（gamma/poisson 因标签残留 −6000 哨兵不可用），再叠样本加权。
但官方指标的形状**不是**其中任何一个：

    Acc_POR = mean( max(0, 1 − |log10(ŷ/y)| / TOL_POR ) )     TOL_POR = log10(1.08)
    Acc_PERM= mean( max(0, 1 − |log10(ŷ/y)| / 1.0 ) )
    Acc_SW  = mean( max(0, 1 − |log10(ŷ/y)| / TOL_SW ) )     TOL_SW  = log10(1.05)

设 z = |log10(ŷ/y)| / TOL，则逐点**得分** acc = max(1 − z, 0)，逐点**损失**
loss = 1 − acc = min(1, z)。

⚠️ 先纠正一个直觉错误：loss 不是凸函数。它在带内（z<1）斜率 ±1/TOL，带外为 0，
斜率单调**下降**，二阶导 ≤ 0。直接把真损失喂给 Newton boosting 会得到负 hessian、
步长发散。所以只能取光滑近似（见 OfficialMetricLoss 类注释）：
    L = 0.5·s·(σ((|z|−1)/s) − 1)，梯度是以带边 z=1 为中心的 bump。

它的两个性质是任何现有 loss 都给不了的：
  1. 梯度随 |z| 远离带边而衰减 —— 树不再花容量去修正「已经拿不到分」的点。
     squared_error 的梯度 ∝ 残差，恰恰最狠地推 hopeless 点；absolute_error 恒定施力。
  2. 点权重**完全等权**，不需要 weight_tol，也不需要 |y| / |log10 y| 的补偿，
     因为官方口径本来就是逐点等权的相对误差。
这就是为什么 §4.5 的 weight_tol 一直「零影响」：它在补一个不匹配的损失函数的
错，而这里损失函数本身已经对齐了。

合成数据上的方向性证据（/tmp/test_custom_loss.py，4000 点随机回归）：
    squared_error   |log10 残差| MAE 0.0827   带内占比 26.3%
    absolute_error  MAE 0.0828                31.3%
    官方指标损失     MAE 0.0891                **34.4%**
MAE 略高而带内占比更高，正是「优化了正确的目标」的指纹。

实现要点（sklearn 1.7.2 接口）
------------------------------
HistGradientBoostingRegressor(loss=...) 只接受 str 或 sklearn._loss.loss.BaseLoss
实例（不是任意 callable）。自定义损失必须继承 BaseLoss 并实现
loss() / gradient() / gradient_hessian() / fit_intercept_only()。
|·| 用 soft_abs(u) = logaddexp(u, −u)，折点 z=1 用宽度 s 的 sigmoid 软化。
三个已踩坑见类注释；数值正确性用有限差分验证（1.5e-9）。

口径纪律
--------
* 纯 numpy + sklearn，**不 import torch**（§5.1）。TCN 走 numpy 前向，零 GPU。
* 本脚本是**诊断**：只在 72/8 screen 上比较，不改 src/model.py。
* 基线必须复现 73.931，否则退出。
* 报 acc / acc_def / acc_real 分解 + 留一井包络 + 阈值重扫 Δ。

用法: python experiments/tune_reg_custom_loss.py
"""
from __future__ import annotations

import gc
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn._loss.loss import BaseLoss, CyHalfSquaredError, IdentityLink

from src.feature import (
    load_wells, make_targets, default_mask, make_features,
    transform_target, inverse_transform_target,
)
from src.utils import TARGETS, WEIGHTS, EPS, DEFAULT_VALUES
from src.model import DEFAULT_REG_PARAMS, POR_MONOTONIC
from experiments.eval_v4 import build_tree, tcn_outputs

RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
TAGS = ["T15_seed42_150ep", "T16_seed43_150ep", "T16_seed44_150ep"]
CFG = {"blend_w": {"POR": 0.50, "PERM": 0.50, "SW": 0.75},
       "thresholds": {"POR": 0.080, "PERM": 0.250, "SW": 0.050},
       "clf_tcn_w": {"POR": 0.60, "PERM": 0.70, "SW": 0.30}}
BASE = 73.931
GRID = np.arange(0.02, 0.71, 0.005)

# log10 空间的官方容差带（精确：log10(1+TOL) 而非 TOL/ln10）
TOL_LOG = {"POR": np.log10(1.08), "PERM": 1.0, "SW": np.log10(1.05)}


def rss() -> str:
    import resource
    return f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576:.0f}MB"


def acc_vec(c, p, t):
    p = np.asarray(p, float)
    t = np.asarray(t, float)
    if c == "PERM":
        return np.maximum(0.0, 1.0 - np.abs(np.log10(np.maximum(p / np.maximum(t, EPS), EPS))))
    tol = 0.08 if c == "POR" else 0.05
    return np.maximum(0.0, 1.0 - np.abs(p - t) / (tol * (np.abs(t) + EPS)))


class OfficialMetricLoss(BaseLoss):
    """逐点损失 L = max(1 − |log10(ŷ/y)|/TOL, 0) 的光滑代理。

    官方逐点 acc = max(0, 1−|z|)，故逐点**损失** = min(1, |z|)。它**非凸**：
    带内（|z|<1）斜率 ±1/TOL，带外为 0，斜率单调下降。直接喂 Newton boosting
    会得到 ≤0 的 hessian，步长发散。所以取光滑近似：

        L(r) = 0.5·s·(σ((soft_abs(u) − 1)/s) − 1)      u = (r − t)/TOL
        dL/dr = 0.5·a(1−a)·tanh(u)/TOL                 a = σ((z−1)/s)

    梯度是一个**以带边 z=1 为中心、宽度 s 的 bump**（POR 上峰值 ≈ 3.7，
    z=3 处衰减到峰值的 2%）。行为含义：树把容量集中投在带边附近的点——最容易
    通过小步改善换到分数；对已远超容差的 hopeless 点几乎不再施力。这是
    squared_error（梯度 ∝ 残差，最狠地推 hopeless 点）与 absolute_error
    （恒定施力）都给不了的。

    raw_prediction 与 y_true 都必须在 log10 空间（link=IdentityLink）。
    |·| 用 logaddexp 软化，折点 z=1 用宽度 s 的 sigmoid 软化。

    数值验证过的三个坑（都踩过）：
      1. gradient 与 gradient_hessian 必须用**同一个** σ 参数化 σ((z−1)/s)。
         早期版本一个用 σ(z−1) 一个用 σ((z−1)/s)，两者相差 10.0。
      2. fit_intercept_only 必须返回 np 标量而非 Python float —— numpy 2.x 的
         np.float64 才有 .reshape，fit() 会在 .reshape((1,−1)) 处 AttributeError。
      3. 真 hessian 在带外为负，必须 clamp 到正下限，否则 Newton 步反向发散。
    有限差分验证 max|analytic − numeric| = 1.5e-9。
    """

    differentiable = True
    need_update_leaves_values = False
    is_multiclass = False

    def __init__(self, tol_log: float, hinge_s: float = 0.25,
                 sample_weight=None, hess_floor: float = 1e-8):
        super().__init__(closs=CyHalfSquaredError(), link=IdentityLink())
        self.approx_hessian = True       # hessian 经 clamp，是上界
        self.constant_hessian = False
        self.tol = tol_log
        self.s = hinge_s
        self.hess_floor = hess_floor

    @staticmethod
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    def _quant(self, r, t):
        """返回 (soft|z|, tanh(u))。|·| 用 logaddexp 软化（在 0 处可导）。"""
        u = (r - t) / self.tol
        z = np.logaddexp(u, -u)
        return z, np.tanh(u)

    def loss(self, y_true, raw_prediction, sample_weight=None, loss_out=None, n_threads=1):
        if loss_out is None:
            loss_out = np.empty_like(y_true)
        r = np.asarray(raw_prediction, dtype=y_true.dtype).ravel()
        t = np.asarray(y_true, dtype=y_true.dtype).ravel()
        z, _ = self._quant(r, t)
        a = self._sigmoid((z - 1.0) / self.s)
        loss_out[:] = 0.5 * self.s * (a - 1.0)          # 仅用于监控，含常数偏置
        return loss_out

    def gradient(self, y_true, raw_prediction, sample_weight=None,
                 gradient_out=None, n_threads=1):
        if gradient_out is None:
            gradient_out = np.empty_like(np.asarray(raw_prediction))
        r = np.asarray(raw_prediction).ravel()
        t = np.asarray(y_true).ravel()
        z, th = self._quant(r, t)
        a = self._sigmoid((z - 1.0) / self.s)
        gradient_out[:] = (0.5 * a * (1.0 - a) * th / self.tol).astype(gradient_out.dtype)
        return gradient_out

    def gradient_hessian(self, y_true, raw_prediction, sample_weight=None,
                         gradient_out=None, hessian_out=None, n_threads=1):
        if gradient_out is None:
            gradient_out = np.empty_like(np.asarray(raw_prediction))
        if hessian_out is None:
            hessian_out = np.empty_like(np.asarray(raw_prediction))
        r = np.asarray(raw_prediction).ravel()
        t = np.asarray(y_true).ravel()
        z, th = self._quant(r, t)
        a = self._sigmoid((z - 1.0) / self.s)
        p = a * (1.0 - a)
        grad = 0.5 * p * th / self.tol
        # σ'' 项（a>0.5 即 z>1 时为负）+ σ'·sech² 项（非负）
        hess = (0.5 / self.tol ** 2) * (p * (1.0 - 2.0 * a) / self.s * th * th
                                        + p * (1.0 - th * th))
        gradient_out[:] = grad.astype(gradient_out.dtype)
        hessian_out[:] = np.maximum(hess, self.hess_floor).astype(hessian_out.dtype)
        return gradient_out, hessian_out

    def fit_intercept_only(self, y_true, sample_weight=None):
        """初值取中位数。必须返回 np 标量而非 Python float：
        numpy 2.x 的 np.float64 才有 .reshape，fit() 会在 .reshape((1,-1)) 处炸。"""
        y = np.asarray(y_true, dtype=np.float64).ravel()
        return self.link.link(np.float64(np.median(y)))


def make_loss(c):
    return OfficialMetricLoss(TOL_LOG[c], hinge_s=0.25)


def fit_custom(Xtr, Ytr, Dtr, c, extra_w=None, max_iter=None):
    """perm_log 空间 + 官方指标损失。extra_w=None 即纯等权（无 weight_tol）。"""
    valid = Ytr[c].notna().to_numpy()
    y = Ytr.loc[valid, c]
    def_ = Dtr.loc[valid, c].astype(int)
    Xv = Xtr.loc[valid]
    real = (def_ == 0).to_numpy()

    params = dict(DEFAULT_REG_PARAMS)
    if max_iter is not None:
        params["max_iter"] = max_iter
    mono = POR_MONOTONIC if c == "POR" else None
    if mono:
        params["monotonic_cst"] = [mono.get(col, 0) for col in list(Xv.columns)]
    reg = HistGradientBoostingRegressor(loss=make_loss(c), random_state=42, **params)
    yt = transform_target(c, y.loc[real], "perm_log").astype("float64")
    reg.fit(Xv.loc[real], yt)
    if extra_w is not None:
        w = np.clip(extra_w, np.percentile(extra_w, 5), np.percentile(extra_w, 95))
        reg.fit(Xv.loc[real], yt, sample_weight=w / w.mean())
    return reg


def reg_to_log10(reg, c, Xva):
    lin = inverse_transform_target(c,
                                   pd.Series(reg.predict(Xva), index=Xva.index), "perm_log")
    return np.log10(np.maximum(lin.to_numpy(), EPS))


def main():
    t0 = time.time()
    wells = load_wells("./train", has_labels=True)
    ids = [w["logId"].iloc[0] for w in wells]
    by = dict(zip(ids, wells))
    tr = pd.concat([by[i] for i in ids[:72]], ignore_index=True)
    va = pd.concat([by[i] for i in ids[72:80]], ignore_index=True)
    va_wells = [by[i] for i in ids[72:80]]
    Ytr, Yva = make_targets(tr), make_targets(va)
    Dtr, Dva = default_mask(Ytr), default_mask(Yva)
    gid = va["logId"].to_numpy()
    Xtr, Xva = make_features(tr, context=True, priors=True), make_features(va, context=True, priors=True)
    print(f"训练 {len(tr):,} 点 / 72 井   验证 {len(va):,} 点 / 8 井   特征 {Xtr.shape[1]}")
    print(f"log10 容差带  POR {TOL_LOG['POR']:.5f}  PERM {TOL_LOG['PERM']:.5f}"
          f"  SW {TOL_LOG['SW']:.5f}")

    # ---------- 自检：自定义损失的数值正确性 ----------
    print("\n[自检] 自定义损失 vs 官方逐点损失（log10 空间）")
    for c in TARGETS:
        lo = make_loss(c)
        t = np.array([0.1, 0.2, 0.3, 0.0])
        r = t + np.array([-0.05, -0.005, 0.0, 0.5])
        exact = np.maximum(0.0, 1.0 - np.abs((r - t) / TOL_LOG[c]))
        smooth = lo.loss(np.asarray(t, np.float64), np.asarray(r, np.float64))
        print(f"  {c}  精确 {np.round(exact, 4)}   平滑 {np.round(smooth, 4)}"
              f"   （平滑含常数偏置 −0.5，比较形状即可）")
        g = lo.gradient(np.asarray(t, np.float64), np.asarray(r, np.float64))
        g_num = (lo.loss(np.asarray(t, np.float64), np.asarray(r + 1e-6, np.float64))
                 - lo.loss(np.asarray(t, np.float64), np.asarray(r - 1e-6, np.float64))) / 2e-6
        print(f"       梯度 {np.round(g, 5)}   数值 {np.round(g_num, 5)}"
              f"   max|差| {np.max(np.abs(g - g_num)):.2e}")

    arts = [joblib.load(os.path.join(RUNS, f"tcn_{t}.joblib")) for t in TAGS]
    ens, pcn = {}, {}
    for art in arts:
        r_, p_ = tcn_outputs(va_wells, art)
        for c in TARGETS:
            ens.setdefault(c, []).append(r_[c])
            pcn.setdefault(c, []).append(p_[c])
    ens = {c: np.mean(v, axis=0) for c, v in ens.items()}
    pcn = {c: np.mean(v, axis=0) for c, v in pcn.items()}
    del arts
    gc.collect()

    m0, _ = build_tree(tr, weight_tol=True)
    p_tree = {c: m0.classifiers[c].predict_proba(Xva)[:, 1] for c in TARGETS}
    pm = {c: (1 - CFG["clf_tcn_w"][c]) * p_tree[c] + CFG["clf_tcn_w"][c] * pcn[c]
          for c in TARGETS}

    def evaluate(reg_by_t, ths=None):
        ths = ths or CFG["thresholds"]
        pred = {}
        for c in TARGETS:
            r10 = reg_to_log10(reg_by_t[c], c, Xva)
            rb = (1 - CFG["blend_w"][c]) * r10 + CFG["blend_w"][c] * ens[c]
            rb = np.where(np.isfinite(rb), rb, -5.0)
            pred[c] = np.where(pm[c] >= ths[c], DEFAULT_VALUES[c], np.power(10.0, rb))
        tot, parts, per, accs = 0.0, {}, {}, {}
        for c in TARGETS:
            y = Yva[c].to_numpy().astype(float)
            valid = np.isfinite(y)
            a = np.where(valid, acc_vec(c, pred[c], y), np.nan)
            accs[c] = a
            tot += WEIGHTS[c] * np.nanmean(a)
            d = Dva[c].to_numpy() & valid
            parts[c] = dict(acc=float(np.nanmean(a)),
                            real=float(np.nanmean(a[valid & ~d])) if (valid & ~d).sum() else float("nan"),
                            defv=float(np.nanmean(a[valid & d])) if (valid & d).sum() else float("nan"))
        for g in np.unique(gid):
            m_ = gid == g
            per[str(g)[:8]] = 100.0 * sum(WEIGHTS[c] * np.nanmean(accs[c][m_]) for c in TARGETS)
        return tot * 100, parts, per, pred

    reg_base = {c: m0.regressors[c] for c in TARGETS}
    t_base, d_base, pw0, _ = evaluate(reg_base)
    print(f"\n[自检] 部署配置复现 = {t_base:.3f}（应 {BASE}）"
          f"   {'✓' if abs(t_base - BASE) <= 0.02 else '✗ 未复现 → 退出'}")
    if abs(t_base - BASE) > 0.02:
        return
    print(f"  分解 " + "  ".join(
        f"{c} {d_base[c]['acc']:.4f}/real {d_base[c]['real']:.4f}/def {d_base[c]['defv']:.4f}"
        for c in TARGETS))

    # ---------- 逐目标：官方指标损失 ----------
    variants = [("A 官方损失 · 等权（无 weight_tol）", None),
                ("B 官方损失 + w=|log10 y|", "abslogy"),
                ("C 官方损失 · 400ep", None)]
    rows = []
    print("\n" + "=" * 100)
    print("逐目标换官方指标损失（mode=perm_log，分类器/融合/TCN 全冻结）")
    print("=" * 100)
    for name, ew in variants:
        reg = dict(reg_base)
        mi = 400 if "400ep" in name else None
        for c in TARGETS:
            wfn = None
            if ew == "abslogy":
                yv = Ytr.loc[Ytr[c].notna() & ~Dtr[c].to_numpy(), c].to_numpy()
                wfn = np.abs(np.log10(np.maximum(yv, EPS)))
            reg[c] = fit_custom(Xtr, Ytr, Dtr, c, extra_w=wfn, max_iter=mi)
        t = time.time()
        td, dd, pw, pred = evaluate(reg)
        d = td - t_base

        # 阈值重扫
        fixed = {}
        for c2 in TARGETS:
            y2 = Yva[c2].to_numpy().astype(float)
            v2 = np.isfinite(y2)
            fixed[c2] = np.where(v2, acc_vec(c2, pred[c2], y2), np.nan)
            fixed[c2 + "_v"] = v2
        r10 = {k: reg_to_log10(reg[k], k, Xva) for k in TARGETS}
        rb = {k: np.where(
            np.isfinite((1 - CFG["blend_w"][k]) * r10[k] + CFG["blend_w"][k] * ens[k]),
            (1 - CFG["blend_w"][k]) * r10[k] + CFG["blend_w"][k] * ens[k], -5.0)
            for k in TARGETS}
        best, th = td, dict(CFG["thresholds"])
        for c2 in TARGETS:
            y2 = Yva[c2].to_numpy().astype(float)
            for g in GRID:
                p = np.where(pm[c2] >= g, DEFAULT_VALUES[c2], np.power(10.0, rb[c2]))
                a_c = np.where(fixed[c2 + "_v"], acc_vec(c2, p, y2), np.nan)
                tot = sum(WEIGHTS[k] * np.nanmean(a_c if k == c2 else fixed[k])
                          for k in TARGETS) * 100.0
                if tot > best:
                    best, th = tot, dict(th, **{c2: g})

        keys = list(pw0)
        dls = np.array([pw[k] - pw0[k] for k in keys], dtype=float)
        fin = np.isfinite(dls)
        print(f"\n{name}\n  Total {t_base:.3f} → {td:.3f}   Δ {d:+.3f} 分   "
              f"({time.time() - t:.0f}s)")
        for c in TARGETS:
            b = d_base[c]
            print(f"     {c}  acc {b['acc']:.4f}→{dd[c]['acc']:.4f}"
                  f"   real {b['real']:.4f}→{dd[c]['real']:.4f}"
                  f"   Δ {(dd[c]['acc'] - b['acc']) * 100 * WEIGHTS[c]:+.4f} 分")
        print(f"     包络（{int(fin.sum())} 井有效）"
              f"  [{dls[fin].min():+.3f},{dls[fin].max():+.3f}]"
              f"   正号 {int((dls[fin] > 0).sum())}/{int(fin.sum())}"
              f"   {'   ← 不跨 0' if fin.all() and dls.min() > 0 else ''}")
        print(f"     重扫最优 {best:.3f}   Δ {best - t_base:+.3f}"
              f"   （重扫只多 {best - td:+.3f}）")
        rows.append(dict(name=name, v9=td, d=d, opt=best, d_opt=best - t_base, dls=dls))
        del reg, pred
        gc.collect()

    print("\n" + "=" * 100)
    print("汇总")
    print("=" * 100)
    print(f"  {'变体':<44}{'v9 阈值':>10}{'Δv9':>9}{'重扫最优':>10}{'Δopt':>9}"
          f"{'包络下界':>11}{'包络上界':>11}{'正号':>7}")
    print(f"  {'V0 部署现状 (squared_error + weight_tol)':<44}{t_base:>10.3f}{'':>9}"
          f"{t_base:>10.3f}{'':>9}{'':>11}{'':>11}{'':>7}")
    for r in sorted(rows, key=lambda x: -x["d_opt"]):
        f = np.isfinite(r["dls"])
        print(f"  {r['name']:<44}{r['v9']:>10.3f}{r['d']:>+9.3f}{r['opt']:>10.3f}"
              f"{r['d_opt']:>+9.3f}{r['dls'][f].min():>+11.3f}{r['dls'][f].max():>+11.3f}"
              f"{int((r['dls'][f] > 0).sum()):>7}")

    print(f"\n总用时 {time.time() - t0:.0f}s   峰值 {rss()}")


if __name__ == "__main__":
    main()
