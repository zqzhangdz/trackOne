"""官方指标作为 GBM 目标函数 —— **修正版**。

上一版 (tune_reg_custom_loss.py) 为什么是错的
---------------------------------------------
官方逐点损失是 L(z) = min(1, |z|)，z = |log10(ŷ/y)|/TOL。它是**硬截断的绝对值**：
  |z| < 1（带内）: 梯度 = ±1/TOL
  |z| > 1（带外）: 梯度 = 0
所以梯度的形状是一个**矩形**：带内幅值恒定 1/TOL，带外为 0，在 z=1 处不连续。

上一版错在平滑对象选反了：它取
    L = 0.5·s·σ((|z|−1)/s)
这是 **acc = max(0, 1−z)（凹函数）** 的光滑近似，不是 loss 的近似。凹函数的梯度是
**bump**（在 z=1 处取峰、两侧都趋于 0），而不是矩形。把这个 bump 当损失梯度喂给
Newton boosting，再配合被 clamp 掉的 hessian，等于让优化器沿一个「只在带边上有力、
带内外都没力」的方向推。结果是树塌到中位数，实测 **−2.206 / 0 井为正**。

那 −2.206 不能用来否决「直接优化官方指标」这个想法 —— 它只否决了那个错误的实现。

正确的实现（本脚本）
--------------------
grad = tanh(u) · w(z) / TOL ,   w(z) = σ((1−z)/s)
  tanh(u)     : 方向，在 u=0 处可导（软化 |·| 的折点）
  w(z)        : 平滑截断，带内 ≈1、带外 ≈0，宽度 s 控制 z=1 处的过渡
hess = [sech²(u)·w − tanh²(u)·p/s] / TOL² ,   p = w(1−w) = σ'((1−z)/s)

⚠️ 本节的手算版**本身是错的**（两个错误：漏了 log 2 偏置、量级算错），下面是数值重算后的
正确版本（TOL=POR=0.033424，TOL²=0.0011171，1/TOL²=895.1）。hess_num = hess·TOL²。

  u (=|z|)  z_soft       w         p=w(1−w)   hess_num      hess      hess·TOL²
    0.0      0.6931    0.77337    0.17527     +0.77337     +692.27    +0.773
    0.5      0.8133    0.67851    0.21813     +0.34729     +310.87    +0.347
    0.8      0.9839    0.51609    0.24974     −0.15196     −136.03    −0.152
    1.0      1.1269    0.37573    0.23456     −0.38640     −345.88    −0.386
    1.1      1.2051    0.30569    0.21224     −0.43422     −388.69    −0.434
    2.0      2.0181    0.01675    0.01647     −0.06003      −53.74    −0.060
    3.0      3.0025    0.00033    0.00033     −0.00131       −1.17    −0.001

两个纠正（上一版手算都错了）：
  * 带边 u=1：手算写「w=0.5、hess = +0.065/TOL²（正）」。实际 z_soft(1)=1.1269≠1，
    所以 w=0.3757，hess = **−0.386/TOL²（负）**。符号就反了。
  * 过渡带 u=2：手算写「hess ≈ −0.015/TOL²，量级只有带内的 1.5%」。实际 **−0.060/TOL²**，
    是带内峰值 692.27 的 7.8%。量级被低估了 4 倍。

关键量（精确解）：
  * **零点偏置**：logaddexp(u,−u) 在 u=0 处 = log 2 = **0.6931，不是 0**。这 0.6931 吃掉
    了 1.0 带宽的 **69.3%**，误差为 0 时截断权重已降到 w=0.773（不是手算的 0.982）。
  * **有效带宽**：z_soft 达到 1.0（截断开始生效）在 **|u|=0.8240**，即只有官方容差带的
    82%。最后 **18%** 的容差带内梯度就在衰减 —— 模型在仍然正确的时候被推着走远。
  * **hessian 零点**：hess 在 **u=0.7066**（z_soft=0.9245）由正转负，即 70.7% 的容差带内
    曲率已经翻转，最负到 −0.434/TOL²（u=1.1）。

结论：hessian 只在很窄的 |u|<0.707 内是正的，之后一路为负。clamp 一个下限会丢掉带外
99.7% 的曲率，Newton 步 grad/hess 从带内的 ~0.1·TOL 爆炸到带外的 ~10⁶·TOL。截断损失 +
Newton boosting 本身病态，不是实现细节能修的。可部署的替代方案见 tune_reg_isolated.py
（公开 API 的 absolute_error + 数据驱动权重）。

loss() 直接放**精确**官方损失 min(1,|z|) —— HistGB 只用它做收敛监控，不用它做
优化。梯度与 hessian 走上面的光滑版本。

口径纪律
--------
* 纯 numpy + sklearn，**不 import torch**（§5.1）。
* 基线必须复现 73.931，否则退出。
* 报 acc/real/def 分解 + 逐井 Δ + LOO jackknife 均值 CI（正确统计量，见
  jackknife_v3.py 的教训：逐井 Δ 的 min/max 不是均值 CI）+ 阈值重扫。
* 与内建 loss 的对照：同一 harness 下 squared_error（部署）、absolute_error、
  以及「absolute_error + 按 pass-1 残差加权」都放进去，避免拿自定义损失的绝对分数
  和别的 harness 比较。

用法: python experiments/tune_reg_official_v2.py
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


def sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def acc_vec(c, p, t):
    p = np.asarray(p, float)
    t = np.asarray(t, float)
    if c == "PERM":
        return np.maximum(0.0, 1.0 - np.abs(np.log10(np.maximum(p / np.maximum(t, EPS), EPS))))
    tol = 0.08 if c == "POR" else 0.05
    return np.maximum(0.0, 1.0 - np.abs(p - t) / (tol * (np.abs(t) + EPS)))


class OfficialMetricLossV2(BaseLoss):
    """L(z) = min(1, |z|) 的光滑化 —— 修正版，梯度是矩形不是 bump。

    见模块 docstring 的符号检查。raw_prediction 与 y_true 都必须在 log10 空间
    （link=IdentityLink）。
    """

    differentiable = True
    need_update_leaves_values = False
    is_multiclass = False

    def __init__(self, tol_log: float, hinge_s: float = 0.25,
                 hess_floor: float = 1e-10):
        super().__init__(closs=CyHalfSquaredError(), link=IdentityLink())
        # hessian 在 z_soft>0.8 后转负（带边峰值 −0.49/TOL²），clamp 到 hess_floor
        # 会丢掉带外 99.7% 的曲率，Newton 步爆炸到 ~1e6·TOL。见模块 docstring 重算表。
        self.approx_hessian = True
        self.constant_hessian = False
        self.tol = tol_log
        self.s = hinge_s
        self.hess_floor = hess_floor

    def _quant(self, r, t):
        """返回 (u, z=soft|u|, tanh(u), w=σ((1−z)/s), p=w(1−w))。"""
        u = (r - t) / self.tol
        z = np.logaddexp(u, -u)          # soft |u|，在 0 处可导
        w = sig((1.0 - z) / self.s)      # 平滑截断：带内 ≈1，带外 ≈0
        return u, z, np.tanh(u), w, w * (1.0 - w)

    def loss(self, y_true, raw_prediction, sample_weight=None, loss_out=None, n_threads=1):
        """精确官方逐点损失，仅供收敛监控。"""
        if loss_out is None:
            loss_out = np.empty_like(y_true)
        r = np.asarray(raw_prediction, dtype=np.float64).ravel()
        t = np.asarray(y_true, dtype=np.float64).ravel()
        u = (r - t) / self.tol
        loss_out[:] = np.minimum(1.0, np.abs(u)).astype(loss_out.dtype)
        return loss_out

    def _grad_hess(self, r, t):
        u, z, th, w, p = self._quant(r, t)
        sech2 = 1.0 - th * th
        grad = th * w / self.tol
        hess = (sech2 * w - th * th * p / self.s) / (self.tol * self.tol)
        return grad, hess

    def gradient(self, y_true, raw_prediction, sample_weight=None,
                 gradient_out=None, n_threads=1):
        if gradient_out is None:
            gradient_out = np.empty_like(np.asarray(raw_prediction))
        r = np.asarray(raw_prediction, dtype=np.float64).ravel()
        t = np.asarray(y_true, dtype=np.float64).ravel()
        grad, _ = self._grad_hess(r, t)
        gradient_out[:] = grad.astype(gradient_out.dtype)
        return gradient_out

    def gradient_hessian(self, y_true, raw_prediction, sample_weight=None,
                         gradient_out=None, hessian_out=None, n_threads=1):
        if gradient_out is None:
            gradient_out = np.empty_like(np.asarray(raw_prediction))
        if hessian_out is None:
            hessian_out = np.empty_like(np.asarray(raw_prediction))
        r = np.asarray(raw_prediction, dtype=np.float64).ravel()
        t = np.asarray(y_true, dtype=np.float64).ravel()
        grad, hess = self._grad_hess(r, t)
        gradient_out[:] = grad.astype(gradient_out.dtype)
        hessian_out[:] = np.maximum(hess, self.hess_floor).astype(hessian_out.dtype)
        return gradient_out, hessian_out

    def fit_intercept_only(self, y_true, sample_weight=None):
        """必须返回 np 标量而非 Python float：numpy 2.x 的 np.float64 才有 .reshape，
        fit() 会在 .reshape((1,−1)) 处 AttributeError。"""
        y = np.asarray(y_true, dtype=np.float64).ravel()
        return self.link.link(np.float64(np.median(y)))


def make_loss(c, s=0.25):
    return OfficialMetricLossV2(TOL_LOG[c], hinge_s=s)


def main():
    t0 = time.time()

    # ---------- 自检 1：梯度形状必须是矩形，不是 bump ----------
    print("=" * 96)
    print("自检 1：梯度形状（上一版在这里错了）")
    print("=" * 96)
    lo = make_loss("POR")
    print(f"  TOL_POR = {TOL_LOG['POR']:.5f}   1/TOL = {1.0 / TOL_LOG['POR']:.2f}")
    print(f"  {'z':>6}{'官方grad|':>11}{'V2 grad|':>11}{'V2 hess':>12}"
          f"{'Newton步/TOL':>13}")
    for z in [0.0, 0.2, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0, 5.0]:
        r = np.array([z * TOL_LOG["POR"]], dtype=np.float64)
        g = float(lo.gradient(np.array([0.0]), r)[0])
        _, h = lo.gradient_hessian(np.array([0.0]), r)
        h = float(h[0])
        exact = 1.0 / TOL_LOG["POR"] if z < 1 else 0.0
        step = abs(g) / max(h, 1e-300)
        print(f"  {z:6.1f}{exact:11.3f}{abs(g):11.3f}{h:12.2f}{step / TOL_LOG['POR']:13.3f}")
    print("  上一版（bump）在 z=0 处梯度≈0、z=1 处取峰 2.67 —— 方向完全反了。")
    print("  V2 应在带内恒定 ≈ 29.9、带外 → 0，Newton 步落在 ~0.4·TOL（带内）。")

    # ---------- 自检 2：数值正确性 ----------
    print("\n" + "=" * 96)
    print("自检 2：梯度数值校验 + hessian 符号")
    print("=" * 96)
    rng = np.random.default_rng(0)
    for c in TARGETS:
        L = make_loss(c)
        n = 4000
        t = rng.normal(0, 0.6, n)
        r = t + rng.normal(0, 0.4, n)
        g = L.gradient(t, r)
        gd = (L.loss(t, r + 1e-7) - L.loss(t, r - 1e-7)) / 2e-7
        # loss() 是精确非光滑版，只在 z≠0 和 z≠1 处可导，取可导子集比较
        z = np.abs((r - t) / TOL_LOG[c])
        good = (np.abs(z - 1.0) > 1e-3) & (np.abs(z) > 1e-3)
        print(f"  {c}  max|grad analytic − numeric|（可导点）"
              f" = {np.max(np.abs(g[good] - gd[good])):.3e}"
              f"   ({good.mean() * 100:.0f}% 点可导)")
        gh, hh = L.gradient_hessian(t, r)
        print(f"       gradient_hessian 与 gradient 一致: {np.max(np.abs(gh - g)):.3e}"
              f"   hess min {hh.min():.3e} max {hh.max():.3e}  全正 {(hh > 0).all()}")
        # Newton 步的尺度检查：应落在带内（0 ~ 1 个 TOL）
        step = np.abs(gh / hh)
        print(f"       Newton 步/TOL: 中位 {np.median(step):.3f}  95%分位 {np.percentile(step, 95):.3f}"
              f"  → 步长合理" if np.median(step) < 1.5 else "  → ⚠️ 步长过大")

    # ---------- 自检 3：合成数据端到端 ----------
    print("\n" + "=" * 96)
    print("自检 3：合成数据端到端（HistGB 接受度 + 带内占比）")
    print("=" * 96)
    rng2 = np.random.default_rng(1)
    X = pd.DataFrame(rng2.normal(size=(6000, 10)))
    ytrue = np.abs(rng2.normal(0.15, 0.05, 6000)) + 0.05
    yt = np.log10(ytrue).astype(np.float64)
    T = TOL_LOG["POR"]
    for name, loss in [("squared_error", "squared_error"),
                       ("absolute_error", "absolute_error"),
                       ("V2(s=0.25)", make_loss("POR", 0.25)),
                       ("V2(s=0.10)", make_loss("POR", 0.10)),
                       ("V2(s=0.50)", make_loss("POR", 0.50))]:
        m = HistGradientBoostingRegressor(loss=loss, max_iter=60, random_state=0).fit(X, yt)
        err = np.abs(m.predict(X) - yt)
        print(f"  {name:<14} |log10残差| MAE {err.mean():.4f}"
              f"  带内 {(err <= T).mean() * 100:5.1f}%"
              f"  |log10残差| 中位 {np.median(err):.4f}"
              f"  z>1 占比 {(err > T).mean() * 100:5.1f}%")

    # ---------- 真实数据 ----------
    wells = load_wells("./train", has_labels=True)
    ids = [w["logId"].iloc[0] for w in wells]
    by = dict(zip(ids, wells))
    tr = pd.concat([by[i] for i in ids[:72]], ignore_index=True)
    va = pd.concat([by[i] for i in ids[72:80]], ignore_index=True)
    va_wells = [by[i] for i in ids[72:80]]
    Ytr, Yva = make_targets(tr), make_targets(va)
    Dtr, Dva = default_mask(Ytr), default_mask(Yva)
    gid = va["logId"].to_numpy()
    key_of = np.array([str(x)[:8] for x in gid])
    Xtr, Xva = make_features(tr, context=True, priors=True), make_features(va, context=True, priors=True)
    print(f"\n训练 {len(tr):,} 点 / 72 井   验证 {len(va):,} 点 / 8 井   特征 {Xtr.shape[1]}")

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

    def reg_to_log10(reg, c, Xva):
        lin = inverse_transform_target(c,
                                       pd.Series(reg.predict(Xva), index=Xva.index), "perm_log")
        return np.log10(np.maximum(lin.to_numpy(), EPS))

    def evaluate(reg_by_t, ths=None):
        ths = ths or CFG["thresholds"]
        pred = {}
        for c in TARGETS:
            r10 = reg_to_log10(reg_by_t[c], c, Xva)
            rb = (1 - CFG["blend_w"][c]) * r10 + CFG["blend_w"][c] * ens[c]
            rb = np.where(np.isfinite(rb), rb, -5.0)
            pred[c] = np.where(pm[c] >= ths[c], DEFAULT_VALUES[c], np.power(10.0, rb))
        accs = {}
        tot = 0.0
        parts = {}
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
        pw = {}
        for k in np.unique(key_of):
            m_ = key_of == k
            pw[k] = 100.0 * sum(WEIGHTS[c] * np.nanmean(accs[c][m_]) for c in TARGETS)
        return tot * 100, parts, pw, pred

    reg_base = {c: m0.regressors[c] for c in TARGETS}
    t_base, d_base, pw0, _ = evaluate(reg_base)
    print(f"\n[自检] 部署配置复现 = {t_base:.3f}（应 {BASE}）"
          f"   {'✓' if abs(t_base - BASE) <= 0.02 else '✗ 未复现 → 退出'}")
    if abs(t_base - BASE) > 0.02:
        return

    def fit_variants(s, wkey=None, max_iter=None):
        """按 (hinge 宽度 s, 样本权重 wkey) 重拟三目标回归器。单次拟合。

        s 为 None 时用内建 loss（此时 wkey 走 sklearn 的 sample_weight）。
        """
        reg = dict(reg_base)
        for c in TARGETS:
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
            L = make_loss(c, s) if s is not None else "squared_error"
            yt = transform_target(c, y.loc[real], "perm_log").astype(np.float64)
            reg[c] = HistGradientBoostingRegressor(loss=L, random_state=42, **params)
            if wkey is None:
                reg[c].fit(Xv.loc[real], yt)
            else:
                yv = y.to_numpy()[real]
                w = np.abs(np.log10(np.maximum(yv, EPS))) if wkey == "abslogy" else np.abs(yv)
                w = np.clip(w, np.percentile(w, 5), np.percentile(w, 95))
                reg[c].fit(Xv.loc[real], yt, sample_weight=w / w.mean())
        return reg

    variants = [
        ("V2 s=0.25 等权",          0.25, None),
        ("V2 s=0.10 更硬截断",      0.10, None),
        ("V2 s=0.50 更软截断",      0.50, None),
        ("V2 s=0.25 + w=|log10 y|", 0.25, "abslogy"),
    ]
    rows = []
    print("\n" + "=" * 96)
    print("官方指标损失 V2（修正版）在 72/8 screen 上")
    print("=" * 96)
    for name, s, ew in variants:
        t = time.time()
        reg = fit_variants(s, wkey=ew)
        td, dd, pw, pred = evaluate(reg)
        d = td - t_base

        # 阈值重扫
        fixed = {}
        for c in TARGETS:
            y2 = Yva[c].to_numpy().astype(float)
            fixed[c] = np.where(np.isfinite(y2), acc_vec(c, pred[c], y2), np.nan)
            fixed[c + "_v"] = np.isfinite(y2)
        r10 = {k: reg_to_log10(reg[k], k, Xva) for k in TARGETS}
        rb = {k: np.where(np.isfinite((1 - CFG["blend_w"][k]) * r10[k] + CFG["blend_w"][k] * ens[k]),
                          (1 - CFG["blend_w"][k]) * r10[k] + CFG["blend_w"][k] * ens[k], -5.0)
              for k in TARGETS}
        best, th = td, dict(CFG["thresholds"])
        for c in TARGETS:
            y2 = Yva[c].to_numpy().astype(float)
            for g_ in GRID:
                p = np.where(pm[c] >= g_, DEFAULT_VALUES[c], np.power(10.0, rb[c]))
                a_c = np.where(fixed[c + "_v"], acc_vec(c, p, y2), np.nan)
                tot = sum(WEIGHTS[k] * np.nanmean(a_c if k == c else fixed[k])
                          for k in TARGETS) * 100.0
                if tot > best:
                    best, th = tot, dict(th, **{c: g_})

        keys = list(pw0)
        dls = np.array([pw[k] - pw0[k] for k in keys], dtype=float)
        fin = np.isfinite(dls)
        # LOO jackknife CI on the mean
        loo = [np.mean(dls[[i for i in range(len(fin)) if fin[i] and i != j]])
               for j in range(len(fin)) if fin[j]]
        loo = np.array(loo)
        from scipy import stats as sps
        tt, pv = sps.ttest_1samp(dls[fin], 0.0)

        print(f"\n{name}\n  Total {t_base:.3f} → {td:.3f}   Δ {d:+.3f} 分   "
              f"({time.time() - t:.0f}s)")
        for c in TARGETS:
            b = d_base[c]
            print(f"     {c}  real {b['real']:.4f}→{dd[c]['real']:.4f}"
                  f"   Δ {(dd[c]['acc'] - b['acc']) * 100 * WEIGHTS[c]:+.4f} 分")
        print(f"     逐井 Δ [{dls[fin].min():+.3f},{dls[fin].max():+.3f}] 正号 "
              f"{int((dls[fin] > 0).sum())}/{int(fin.sum())}")
        print(f"     LOO jackknife 均值 CI [{loo.min():+.3f},{loo.max():+.3f}]"
              f"   {'不跨 0' if loo.min() > 0 else '跨 0'}"
              f"   t={tt:.2f} p={pv:.3f}")
        print(f"     重扫最优 {best:.3f}  Δ {best - t_base:+.3f}"
              f"（重扫只多 {best - td:+.3f}）")
        rows.append(dict(name=name, d=d, opt=best - t_base, dls=dls,
                         ci=(float(loo.min()), float(loo.max())), p=pv))
        del reg, pred
        gc.collect()

    print("\n" + "=" * 96)
    print("汇总")
    print("=" * 96)
    print(f"  {'变体':<32}{'Δv9阈值':>10}{'重扫最优':>10}{'CI下界':>10}{'CI上界':>10}"
          f"{'p值':>8}{'正号':>7}")
    print(f"  {'V0 部署现状':<32}{0.0:>10.3f}{0.0:>10.3f}{'':>10}{'':>10}{'':>8}{'':>7}")
    for r in sorted(rows, key=lambda x: -x["d"]):
        f = np.isfinite(r["dls"])
        print(f"  {r['name']:<32}{r['d']:>+10.3f}{r['opt']:>+10.3f}"
              f"{r['ci'][0]:>+10.3f}{r['ci'][1]:>+10.3f}{r['p']:>8.3f}"
              f"{int((r['dls'][f] > 0).sum()):>7}")

    print(f"\n总用时 {time.time() - t0:.0f}s   峰值 {rss()}")


if __name__ == "__main__":
    main()
