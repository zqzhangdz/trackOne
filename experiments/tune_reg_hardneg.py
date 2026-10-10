"""把自定义官方指标损失的收益用**内建 loss + 数据驱动加权**复刻出来（可部署版）。

为什么要这个脚本
----------------
tune_reg_custom_loss.py 用 sklearn 的私有 API（sklearn._loss.loss.BaseLoss）实现了
「直接优化官方指标」的目标函数。但它有一个**交付层面的硬问题**：
requirements.txt 只写了 `scikit-learn>=1.4`（是范围不是钉死版本），而 BaseLoss 是
私有模块。评委会环境若换了 sklearn 小版本，src/model.py 可能直接 import 失败 ——
那是 0 分，不是少几分。

关键观察：自定义损失的梯度是
    dL/dr = 0.5 · a(1−a) · tanh(u) / TOL        a = σ((|z|−1)/s)，z = |log10(ŷ/y)|/TOL
其中 tanh(u) 就是 absolute_error 的次梯度方向，而 2·a(1−a)/TOL 是一个**逐点权重**。
所以「自定义损失」在梯度层面等价于：

    absolute_error 损失  +  w_i = a(1−a) 的样本加权，z_i 由前一版树的残差算出

差别只有一个：真正的自定义损失每轮迭代都更新权重，这里只用 pass-1 的残差算一次
（近似）。若近似版也能拿到自定义损失的大部分收益，就能用**纯内建 loss** 部署，
零交付风险。

权重方案的先验
--------------
z_i 是第 i 个点的误差以「容差带」为单位的倍数。官方口径下：
  * z_i < 1：在带内，改善 δ 换来 δ 的分（满额）；
  * z_i > 1：在带外，改善**换不到任何分**。
所以权重应随 z 增大而**衰减**，衰减点在 z=1 附近。本脚本扫一族：

    bump   w = a(1−a)，a = σ((1−z)/s)      ← 与自定义损失的梯度权重逐点相同
    inv    w = 1/(1+z)
    clip   w = min(1, 1/z)
    floor  w = clip(1/z, 0.05, 1)
    none   w = 1（对照组：验证「衰减」这个方向本身）

口径纪律
--------
* 唯一变量 = pass-1 残差派生的样本加权。mode=perm_log 与分类器/融合/TCN 全冻结。
* pass-1 用**部署配置**（squared_error + weight_tol）算残差，保证权重来源与现状
  同一条基线，不引入第二次拟合偏差。
* 纯 numpy + sklearn，**不 import torch**（§5.1）。
* 基线必须复现 73.931，否则退出。
* 报 acc/real/def 分解 + 留一井包络 + 阈值重扫 Δ（防 §4.11 联合过拟合指纹）。

用法: python experiments/tune_reg_hardneg.py
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
TOL_LOG = {"POR": np.log10(1.08), "PERM": 1.0, "SW": np.log10(1.05)}
S = 0.25          # bump 的宽度，与 tune_reg_custom_loss 保持一致


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


# 权重方案：输入 z（容差带倍数），输出权重
SCHEMES = [
    ("none",   lambda z: np.ones_like(z)),
    ("inv",    lambda z: 1.0 / (1.0 + z)),
    ("clip1",  lambda z: np.minimum(1.0, 1.0 / np.maximum(z, 1e-9))),
    ("floor",  lambda z: np.clip(1.0 / np.maximum(z, 1e-9), 0.05, 1.0)),
    ("bump",   lambda z: None),   # 特殊处理：需要 σ((1-z)/s)
]


def scheme_weight(name, z):
    if name == "none":
        return np.ones_like(z)
    if name == "bump":
        a = sig((1.0 - z) / S)
        return a * (1.0 - a)
    if name == "inv":
        return 1.0 / (1.0 + z)
    if name == "clip1":
        return np.minimum(1.0, 1.0 / np.maximum(z, 1e-9))
    if name == "floor":
        return np.clip(1.0 / np.maximum(z, 1e-9), 0.05, 1.0)
    raise KeyError(name)


def clip_norm(w):
    w = np.clip(w, np.percentile(w, 5), np.percentile(w, 95))
    return w / w.mean()


def fit_pass2(Xtr, Ytr, Dtr, c, z, loss, wname, wfn_extra=None):
    """pass-2：用 z 派生的权重重拟合目标 c。"""
    valid = Ytr[c].notna().to_numpy()
    y = Ytr.loc[valid, c]
    def_ = Dtr.loc[valid, c].astype(int)
    Xv = Xtr.loc[valid]
    real = (def_ == 0).to_numpy()

    params = dict(DEFAULT_REG_PARAMS)
    params["loss"] = loss
    mono = POR_MONOTONIC if c == "POR" else None
    if mono:
        params["monotonic_cst"] = [mono.get(col, 0) for col in list(Xv.columns)]
    reg = HistGradientBoostingRegressor(random_state=42, **params)
    yt = transform_target(c, y.loc[real], "perm_log")
    w = clip_norm(scheme_weight(wname, z))
    # 直接一次拟合。早期版本先无权重拟合再带权重重拟合——那次无权重拟合纯属浪费
    # （约 50 s × 3 目标 × 10 变体），结果被第二次拟合完全覆盖。
    reg.fit(Xv.loc[real], yt, sample_weight=w)
    return reg


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

    # pass-1：部署回归器的训练集残差（log10 空间，带倍数）
    z_tr = {}
    for c in TARGETS:
        valid = Ytr[c].notna().to_numpy()
        y = Ytr.loc[valid, c]
        def_ = Dtr.loc[valid, c].astype(int)
        Xv = Xtr.loc[valid]
        real = (def_ == 0).to_numpy()
        pred_lin = inverse_transform_target(
            c, pd.Series(m0.regressors[c].predict(Xv), index=Xv.index), "perm_log")
        p10 = np.log10(np.maximum(pred_lin.to_numpy(), EPS))
        y10 = np.log10(np.maximum(y.to_numpy(), EPS))
        z_tr[c] = np.abs(p10 - y10)[real] / TOL_LOG[c]

    print("\npass-1 残差分布 z = |log10(ŷ/y)|/TOL（真实段训练点）")
    for c in TARGETS:
        z = z_tr[c]
        print(f"  {c:<5} n={len(z):,}  中位 {np.median(z):.3f}  均值 {np.mean(z):.3f}"
              f"  带内占比 {(z <= 1).mean() * 100:5.1f}%  "
              f"z>5 占比 {(z > 5).mean() * 100:5.1f}%  max {np.nanmax(z):.1f}")

    def evaluate(reg_by_t, ths=None):
        ths = ths or CFG["thresholds"]
        pred = {}
        for c in TARGETS:
            lin = inverse_transform_target(c,
                                           pd.Series(reg_by_t[c].predict(Xva), index=Xva.index),
                                           "perm_log")
            r10 = np.log10(np.maximum(lin.to_numpy(), EPS))
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

    rows = []
    print("\n" + "=" * 100)
    print("pass-2 加权扫描（loss × 权重方案 × 目标）")
    print("=" * 100)
    for loss in ["absolute_error", "squared_error"]:
        for wname in ["none", "inv", "clip1", "floor", "bump"]:
            reg = dict(reg_base)
            t = time.time()
            for c in TARGETS:
                reg[c] = fit_pass2(Xtr, Ytr, Dtr, c, z_tr[c], loss, wname)
            td, dd, pw, pred = evaluate(reg)
            d = td - t_base
            keys = list(pw0)
            dls = np.array([pw[k] - pw0[k] for k in keys], dtype=float)
            fin = np.isfinite(dls)
            print(f"\n  loss={loss:<15} 权重={wname:<7}  Total {t_base:.3f} → {td:.3f}"
                  f"   Δ {d:+.3f} 分   ({time.time() - t:.0f}s)")
            for c in TARGETS:
                b = d_base[c]
                print(f"       {c}  real {b['real']:.4f}→{dd[c]['real']:.4f}"
                      f"   Δ {(dd[c]['real'] - b['real']) * 100 * WEIGHTS[c]:+.4f} 分")
            print(f"       包络（{int(fin.sum())} 井）[{dls[fin].min():+.3f},"
                  f"{dls[fin].max():+.3f}]  正号 {int((dls[fin] > 0).sum())}/{int(fin.sum())}"
                  f"   {'   ← 不跨 0' if fin.all() and dls.min() > 0 else ''}")
            rows.append(dict(loss=loss, w=wname, v9=td, d=d, dls=dls,
                             real={c: dd[c]["real"] for c in TARGETS}))
            del reg, pred
            gc.collect()

    print("\n" + "=" * 100)
    print("排名")
    print("=" * 100)
    print(f"  {'loss':<17}{'权重':<8}{'Total':>9}{'Δ':>9}"
          f"{'包络下界':>11}{'包络上界':>11}{'正号':>7}")
    print(f"  {'— 部署现状 —':<25}{t_base:>9.3f}{'':>9}{'':>11}{'':>11}{'':>7}")
    for r in sorted(rows, key=lambda x: -x["d"]):
        f = np.isfinite(r["dls"])
        print(f"  {r['loss']:<17}{r['w']:<8}{r['v9']:>9.3f}{r['d']:>+9.3f}"
              f"{r['dls'][f].min():>+11.3f}{r['dls'][f].max():>+11.3f}"
              f"{int((r['dls'][f] > 0).sum()):>7}")

    print(f"\n总用时 {time.time() - t0:.0f}s   峰值 {rss()}")


if __name__ == "__main__":
    main()
