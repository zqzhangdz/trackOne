"""逐目标**孤立**残差加权：只动 POR，PERM / SW 完全冻结在部署配置。

tune_reg_hardneg.py 的结论与它留下的缺口
-----------------------------------------
hardneg 把 (loss, 权重) 同时施加到三目标上，Total +0.043（`absolute_error × inv`）。
但它的**逐目标 Δ 打印有 2.5× 的缩放错误**：它算的是
`(Δacc_real × 100 × WEIGHTS)`，漏掉了分类器判成「真实段」的点的占比 f。回归器只影响
被分类器判为真实段的点，所以真实分数贡献是 `Δacc_real × f × 100 × WEIGHTS`。

f 由本脚本 `none` 变体反推 = 0.403（Δreal −0.0030 → Δacc −0.00121）。修正后的
hardneg `absolute_error × inv` 逐目标分解（相加精确等于 Total +0.043）：
  POR   +0.1593/2.5 = **+0.064**   （权重=inv）   +0.1948/2.5 = **+0.078**（bump）
  PERM  −0.0573/2.5 = **−0.023**
  SW    +0.0048/2.5 = **+0.002**
  合计  +0.064 − 0.023 + 0.002 = **+0.043** ✓

所以 hardneg 的 Total 数字一直是**对的**（它是从全量加权均值直接算的），错的只是
逐目标分解的打印。真实收益比看起来小 2.5 倍，但方向没变：POR 的收益是真的
（+0.06 ~ +0.08），被 PERM 的 loss 换价（−0.023）吃掉了近一半。

PERM 的负值**不是加权造成的**：PERM 的 pass-1 残差 99.8% 在带内（中位 z=0.074），
所以 w = min(1, 1/z) = 1、w = clip(1/z, 0.05, 1) = 1、w = none 全部退化为恒等权重。
证据是 PERM Δ 在 none / clip1 / floor 三个变体上**完全相同**。
那部分是 loss 从 squared_error 换成 absolute_error 的纯代价。

结论：只动 POR 应该把 PERM 那 −0.023 的代价拿回来，直接读出 POR 的
+0.064（inv）/ +0.078（bump）。但这仍然远低于 §4.15 的 +0.3 screen 门槛。

本脚本做什么
------------
只动 POR 的回归器，PERM 与 SW 直接沿用 build_tree 的部署回归器（零重拟合）。
这样 POR 的收益不被 PERM 的 loss 代价稀释，可以直接读出来。

⚠️ 但必须过 jackknife。jackknife_v3.py 证明 V3 的 +0.050 在 LOO 均值 CI 上是
[−0.042, +0.165]、p=0.59、且只有 2/7 井三目标同号 —— 单井波动（±0.83）远大于
总收益（+0.05）。POR 的 +0.1593 同样是 7 口井的均值，必须用 LOO 均值 CI 判显著性，
不能只看单井 Δ 的 min/max（那永远跨 0）。

口径纪律
--------
* 唯一变量 = POR 回归器的 (loss, 残差权重)。PERM/SW 回归器、三目标分类器、
  融合权重、TCN 全冻结。
* 纯 numpy + sklearn，**不 import torch**（§5.1）。TCN 走 numpy 前向，零 GPU。
* 基线必须复现 73.931，否则退出。
* 报 acc/real/def 分解 + 逐井 Δ + **LOO jackknife 均值 CI** + t 检验 + 阈值重扫。

用法: python experiments/tune_reg_isolated.py
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
from scipy import stats as sps
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
import sys
TARGET = sys.argv[1] if len(sys.argv) > 1 else "POR"   # 只动这一个目标
S = 0.25


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


def weight_from_z(name, z):
    """z = pass-1 残差 / TOL（带倍数）。权重随 z 衰减。"""
    if name == "none":
        return np.ones_like(z)
    if name == "inv":
        return 1.0 / (1.0 + z)
    if name == "clip1":
        return np.minimum(1.0, 1.0 / np.maximum(z, 1e-9))
    if name == "floor":
        return np.clip(1.0 / np.maximum(z, 1e-9), 0.05, 1.0)
    if name == "bump":
        a = sig((1.0 - z) / S)
        return a * (1.0 - a)
    if name == "inv2":
        return 1.0 / (1.0 + z) ** 2
    raise KeyError(name)


def clip_norm(w):
    w = np.clip(w, np.percentile(w, 5), np.percentile(w, 95))
    return w / w.mean()


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
    key_of = np.array([str(x)[:8] for x in gid])
    Xtr, Xva = make_features(tr, context=True, priors=True), make_features(va, context=True, priors=True)
    print(f"训练 {len(tr):,} 点 / 72 井   验证 {len(va):,} 点 / 8 井   特征 {Xtr.shape[1]}")
    print(f"唯一变量 = {TARGET} 回归器的 (loss, 残差权重)；"
          f"{' / '.join(c for c in TARGETS if c != TARGET)} 与分类器/TCN 全冻结")

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

    # ---------- pass-1 残差（部署 POR 回归器，训练集真实段） ----------
    valid = Ytr[TARGET].notna().to_numpy()
    y = Ytr.loc[valid, TARGET]
    def_ = Dtr.loc[valid, TARGET].astype(int)
    Xv = Xtr.loc[valid]
    real = (def_ == 0).to_numpy()
    pred_lin = inverse_transform_target(
        TARGET, pd.Series(m0.regressors[TARGET].predict(Xv), index=Xv.index), "perm_log")
    p10 = np.log10(np.maximum(pred_lin.to_numpy(), EPS))
    y10 = np.log10(np.maximum(y.to_numpy(), EPS))
    z = np.abs(p10 - y10)[real] / TOL_LOG[TARGET]
    print(f"\npass-1 {TARGET} 残差（训练集真实段）  n={len(z):,}"
          f"  中位 z {np.median(z):.3f}  均值 {np.mean(z):.3f}"
          f"  带内 {(z <= 1).mean() * 100:5.1f}%  z>5 占比 {(z > 5).mean() * 100:5.1f}%")
    print("  各权重方案的分布（clip_norm 前）:")
    for wn in ["none", "inv", "inv2", "clip1", "floor", "bump"]:
        w = weight_from_z(wn, z)
        print(f"    {wn:<7} 中位 {np.median(w):.4f}  p5 {np.percentile(w, 5):.4f}"
              f"  p95 {np.percentile(w, 95):.4f}  min/max {w.min():.4f}/{w.max():.4f}"
              f"  z>5 点得权重 {w[z > 5].mean() if (z > 5).any() else float('nan'):.4f}")

    def evaluate(reg_by_t, ths=None):
        ths = ths or CFG["thresholds"]
        pred, accs = {}, {}
        for c in TARGETS:
            lin = inverse_transform_target(c,
                                           pd.Series(reg_by_t[c].predict(Xva), index=Xva.index),
                                           "perm_log")
            r10 = np.log10(np.maximum(lin.to_numpy(), EPS))
            rb = (1 - CFG["blend_w"][c]) * r10 + CFG["blend_w"][c] * ens[c]
            rb = np.where(np.isfinite(rb), rb, -5.0)
            pred[c] = np.where(pm[c] >= ths[c], DEFAULT_VALUES[c], np.power(10.0, rb))
        tot = 0.0
        parts = {}
        for c in TARGETS:
            yv = Yva[c].to_numpy().astype(float)
            vf = np.isfinite(yv)
            a = np.where(vf, acc_vec(c, pred[c], yv), np.nan)
            accs[c] = a
            tot += WEIGHTS[c] * np.nanmean(a)
            d = Dva[c].to_numpy() & vf
            parts[c] = dict(acc=float(np.nanmean(a)),
                            real=float(np.nanmean(a[vf & ~d])) if (vf & ~d).sum() else float("nan"),
                            defv=float(np.nanmean(a[vf & d])) if (vf & d).sum() else float("nan"))
        pw = {}
        for k in np.unique(key_of):
            mk = key_of == k
            pw[k] = 100.0 * sum(WEIGHTS[c] * np.nanmean(accs[c][mk]) for c in TARGETS)
        return tot * 100, parts, pw, pred

    reg_base = {c: m0.regressors[c] for c in TARGETS}
    t_base, d_base, pw0, _ = evaluate(reg_base)
    print(f"\n[自检] 部署配置复现 = {t_base:.3f}（应 {BASE}）"
          f"   {'✓' if abs(t_base - BASE) <= 0.02 else '✗ 未复现 → 退出'}")
    if abs(t_base - BASE) > 0.02:
        return
    print(f"  分解 " + "  ".join(
        f"{c} acc {d_base[c]['acc']:.4f}/real {d_base[c]['real']:.4f}/def {d_base[c]['defv']:.4f}"
        for c in TARGETS))

    def fit_por(loss, wname):
        params = dict(DEFAULT_REG_PARAMS)
        params["loss"] = loss
        mono = POR_MONOTONIC if TARGET == "POR" else None
        if mono:
            params["monotonic_cst"] = [mono.get(col, 0) for col in list(Xv.columns)]
        yt = transform_target(TARGET, y.loc[real], "perm_log")
        w = clip_norm(weight_from_z(wname, z))
        reg = HistGradientBoostingRegressor(random_state=42, **params)
        reg.fit(Xv.loc[real], yt, sample_weight=w)
        return reg

    rows = []
    configs = [(l, wn) for l in ["absolute_error", "squared_error"]
               for wn in ["none", "inv", "inv2", "clip1", "floor", "bump"]]
    print("\n" + "=" * 100)
    print(f"{TARGET} 孤立残差加权扫描（PERM/SW 冻结于部署）")
    print("=" * 100)
    for loss, wname in configs:
        t = time.time()
        reg = dict(reg_base)
        reg[TARGET] = fit_por(loss, wname)
        td, dd, pw, pred = evaluate(reg)
        d = td - t_base

        keys = list(pw0)
        dls = np.array([pw[k] - pw0[k] for k in keys], dtype=float)
        fin = np.isfinite(dls)
        loo = np.array([np.mean(dls[[i for i in range(len(fin)) if fin[i] and i != j]])
                        for j in range(len(fin)) if fin[j]])
        tt, pv = sps.ttest_1samp(dls[fin], 0.0)

        # 阈值重扫（只扫 POR）
        fixed = {}
        for c in TARGETS:
            y2 = Yva[c].to_numpy().astype(float)
            fixed[c] = np.where(np.isfinite(y2), acc_vec(c, pred[c], y2), np.nan)
            fixed[c + "_v"] = np.isfinite(y2)
        lin = inverse_transform_target(TARGET,
                                       pd.Series(reg[TARGET].predict(Xva), index=Xva.index), "perm_log")
        r10 = np.log10(np.maximum(lin.to_numpy(), EPS))
        rb = np.where(np.isfinite((1 - CFG["blend_w"][TARGET]) * r10
                                  + CFG["blend_w"][TARGET] * ens[TARGET]),
                      (1 - CFG["blend_w"][TARGET]) * r10 + CFG["blend_w"][TARGET] * ens[TARGET],
                      -5.0)
        best, th = td, CFG["thresholds"][TARGET]
        y2 = Yva[TARGET].to_numpy().astype(float)
        for g in GRID:
            p = np.where(pm[TARGET] >= g, DEFAULT_VALUES[TARGET], np.power(10.0, rb))
            a_c = np.where(fixed[TARGET + "_v"], acc_vec(TARGET, p, y2), np.nan)
            tot = sum(WEIGHTS[k] * np.nanmean(a_c if k == TARGET else fixed[k])
                      for k in TARGETS) * 100.0
            if tot > best:
                best, th = tot, g

        b = d_base[TARGET]
        print(f"\n  {loss:<16} 权重 {wname:<7}  Total {t_base:.3f} → {td:.3f}   Δ {d:+.3f} 分"
              f"   ({time.time() - t:.0f}s)")
        print(f"       {TARGET} real {b['real']:.4f}→{dd[TARGET]['real']:.4f}"
              f"   Δ {(dd[TARGET]['acc'] - b['acc']) * 100 * WEIGHTS[TARGET]:+.4f} 分")
        print(f"       逐井 Δ [{dls[fin].min():+.3f},{dls[fin].max():+.3f}] 正号 "
              f"{int((dls[fin] > 0).sum())}/{int(fin.sum())}")
        print(f"       LOO 均值 CI [{loo.min():+.3f},{loo.max():+.3f}]"
              f"   {'★ 不跨 0' if loo.min() > 0 else '跨 0'}"
              f"   t={tt:.2f} p={pv:.3f} {'★' if pv < 0.05 else ''}")
        print(f"       重扫最优 {best:.3f}  Δ {best - t_base:+.3f}（阈值 {th:.3f}）")
        rows.append(dict(loss=loss, w=wname, d=d, dls=dls,
                         ci=(float(loo.min()), float(loo.max())), p=float(pv),
                         opt=best - t_base, th=th))
        del reg, pred
        gc.collect()

    print("\n" + "=" * 100)
    print("汇总（按 LOO CI 下界排序 —— 这是唯一能判显著性的统计量）")
    print("=" * 100)
    print(f"  {'loss':<18}{'权重':<8}{'Δv9':>9}{'CI下界':>10}{'CI上界':>10}"
          f"{'p值':>8}{'正号':>7}{'重扫Δ':>9}")
    print(f"  {'— 部署现状 —':<26}{0.0:>9.3f}{'':>10}{'':>10}{'':>8}{'':>7}{'':>9}")
    for r in sorted(rows, key=lambda x: -x["ci"][0]):
        f = np.isfinite(r["dls"])
        star = "★" if r["ci"][0] > 0 else " "
        print(f"  {star}{r['loss']:<17}{r['w']:<8}{r['d']:>+9.3f}{r['ci'][0]:>+10.3f}"
              f"{r['ci'][1]:>+10.3f}{r['p']:>8.3f}{int((r['dls'][f] > 0).sum()):>7}"
              f"{r['opt']:>+9.3f}")

    print(f"\n总用时 {time.time() - t0:.0f}s   峰值 {rss()}")


if __name__ == "__main__":
    main()
