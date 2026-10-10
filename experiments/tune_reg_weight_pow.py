"""POR / SW 样本加权指数扫描：把「|y| → |log10 y|」做成连续轴。

为什么这个脚本存在
-----------------
tune_reg_objective.py 的 V3（三目标全 all_log+MAE+w=|log10 y|）拿到 +0.050，
V4（同损失、去掉加权）是 −0.050，据此猜「加权方向翻转」是增益主体。
但复核 train.py:281 后发现有更精确的说法：

    部署现状：  PERM 用 |log10 y|，POR / SW 用 |y|

所以 V3 相对基线的**真实差异**只有 POR 与 SW 两处加权——PERM 根本没变。
「POR / SW 该往哪个方向加权」这条轴从未被单独测过：它一直被包在 weight_tol
这个整体开关里（§4.5 只记录了整体开关对官方 acc「零影响」，那是整体，不是
逐目标拆开的）。

机理（先验，本脚本检验它是否成立）
----------------------------------
官方 POR/SW 是**相对**误差 |ŷ−y|/(TOL·|y|)，逐点等权。而 squared_error 在原始
y 上的梯度 ∝ 2r，同一个相对误差下大 y 点的梯度绝对值远大；再叠 w=|y| 是乘性
再放大一次，等于把模型往「大 y 点」推——与官方口径正好相反。把 w 换成
|log10 y| 后权重只在 0..2 间慢变，近似把每个点的影响在 log 尺度上拉平，
更接近逐点等权的相对误差。若这个解释对，增益应随「加大 y 权重的程度」
单调递减；若只是巧合，曲线会无规律。

设计
----
perm_log + squared_error + 分类器 + 融合权重 + TCN 全冻结，只扫一个目标的
样本加权。PERM 固定为部署的 |log10 y|（其回归器直接取自 build_tree，只拟一次），
只动 POR 与 SW——每个变体只拟 1 个目标，约 90 s / 变体。

口径纪律
--------
* 唯一变量 = 一个目标的样本加权函数。
* 纯 numpy + sklearn，**不 import torch**（§5.1）。TCN 走 numpy 前向，零 GPU。
* 基线必须复现 73.931，否则退出。
* 报 acc / acc_def / acc_real 分解 + 留一井包络（8 井是实验单元；行级
  bootstrap 无效，POR lag-1 自相关 0.967）+ 阈值重扫 Δ（防 §4.11 联合过拟合）。

用法: python experiments/tune_reg_weight_pow.py
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

# 加权方案（y = 真实段的一维 numpy）。名字后缀 _deploy 标出部署现状。
WEIGHT_SCHEMES = [
    ("p0_uniform",   lambda y: np.ones_like(y)),
    ("py05",         lambda y: np.abs(y) ** 0.5),
    ("py1_deploy",   lambda y: np.abs(y)),
    ("logy05",       lambda y: np.abs(np.log10(np.maximum(y, EPS))) ** 0.5),
    ("logy1",        lambda y: np.abs(np.log10(np.maximum(y, EPS)))),
    ("logy15",       lambda y: np.abs(np.log10(np.maximum(y, EPS))) ** 1.5),
]
SWEEP_TARGETS = ["POR", "SW"]   # PERM 已是 logy1，不动


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


def clip_norm(w: np.ndarray) -> np.ndarray:
    """与 train.py:282-283 完全一致：截断到 [p5, p95] 后按均值归一。"""
    w = np.clip(w, np.percentile(w, 5), np.percentile(w, 95))
    return w / w.mean()


def fit_one(Xtr, Ytr, Dtr, c, wfn):
    """只拟合目标 c 的回归器：perm_log + squared_error + wfn。"""
    valid = Ytr[c].notna().to_numpy()
    y = Ytr.loc[valid, c]
    def_ = Dtr.loc[valid, c].astype(int)
    Xv = Xtr.loc[valid]
    real = (def_ == 0).to_numpy()

    params = dict(DEFAULT_REG_PARAMS)
    mono = POR_MONOTONIC if c == "POR" else None
    if mono:
        params["monotonic_cst"] = [mono.get(col, 0) for col in list(Xv.columns)]
    reg = HistGradientBoostingRegressor(loss="squared_error", random_state=42, **params)
    yt = transform_target(c, y.loc[real], "perm_log")
    reg.fit(Xv.loc[real], yt)
    w = clip_norm(wfn(y.to_numpy()[real]))
    reg.fit(Xv.loc[real], yt, sample_weight=w)
    return reg


def reg_to_log10(reg, c, Xva):
    """树预测 → log10 空间（perm_log 逆变换后取对数）。"""
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

    arts = [joblib.load(os.path.join(RUNS, f"tcn_{t}.joblib")) for t in TAGS]
    ens, pcn = {}, {}
    for art in arts:
        r, p = tcn_outputs(va_wells, art)
        for c in TARGETS:
            ens.setdefault(c, []).append(r[c])
            pcn.setdefault(c, []).append(p[c])
    ens = {c: np.mean(v, axis=0) for c, v in ens.items()}
    pcn = {c: np.mean(v, axis=0) for c, v in pcn.items()}
    del arts
    gc.collect()

    # 部署分段器 + 部署回归器（全部冻结自 build_tree）
    m0, _ = build_tree(tr, weight_tol=True)
    p_tree = {c: m0.classifiers[c].predict_proba(Xva)[:, 1] for c in TARGETS}
    pm = {c: (1 - CFG["clf_tcn_w"][c]) * p_tree[c] + CFG["clf_tcn_w"][c] * pcn[c]
          for c in TARGETS}

    def evaluate(reg_by_t, ths=None):
        """返回 (Total, 分解, 逐井 Total, {c: 原始量纲预测})。"""
        ths = ths or CFG["thresholds"]
        pred = {}
        for c in TARGETS:
            r10 = reg_to_log10(reg_by_t[c], c, Xva)
            rb = (1 - CFG["blend_w"][c]) * r10 + CFG["blend_w"][c] * ens[c]
            rb = np.where(np.isfinite(rb), rb, -5.0)
            pred[c] = np.where(pm[c] >= ths[c], DEFAULT_VALUES[c], np.power(10.0, rb))

        tot, parts, per = 0.0, {}, {}
        accs = {}
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

    # ---------- 基线 ----------
    reg_base = {c: m0.regressors[c] for c in TARGETS}
    t_base, d_base, pw0, _ = evaluate(reg_base)
    print(f"\n[自检] 部署配置复现 = {t_base:.3f}（应 {BASE}）"
          f"   {'✓' if abs(t_base - BASE) <= 0.02 else '✗ 未复现 → 退出'}")
    if abs(t_base - BASE) > 0.02:
        return
    print(f"  分解 " + "  ".join(
        f"{c} {d_base[c]['acc']:.4f}/real {d_base[c]['real']:.4f}/def {d_base[c]['defv']:.4f}"
        for c in TARGETS))
    print(f"  逐井 " + "  ".join(f"{k[:6]} {v:.3f}" for k, v in pw0.items()))

    # ---------- 加权扫描 ----------
    print("\n" + "=" * 100)
    print("加权扫描（perm_log + squared_error 冻结，只改一个目标的样本加权）")
    print("=" * 100)
    rows = []
    for c in SWEEP_TARGETS:
        for key, wfn in WEIGHT_SCHEMES:
            t = time.time()
            reg = dict(reg_base)
            if key != "py1_deploy":
                reg[c] = fit_one(Xtr, Ytr, Dtr, c, wfn)
            td, dd, pw, pred = evaluate(reg)
            d = td - t_base
            # 阈值重扫（只扫本目标；其他目标在 v9 阈值下预先算好 acc 向量）
            fixed = {}
            for c2 in TARGETS:
                y2 = Yva[c2].to_numpy().astype(float)
                valid2 = np.isfinite(y2)
                fixed[c2] = np.where(valid2, acc_vec(c2, pred[c2], y2), np.nan)
                fixed[c2 + "_v"] = valid2
            r10 = {k2: reg_to_log10(reg[k2], k2, Xva) for k2 in TARGETS}
            rb = {k2: np.where(
                np.isfinite((1 - CFG["blend_w"][k2]) * r10[k2] + CFG["blend_w"][k2] * ens[k2]),
                (1 - CFG["blend_w"][k2]) * r10[k2] + CFG["blend_w"][k2] * ens[k2], -5.0)
                for k2 in TARGETS}
            best, th = td, dict(CFG["thresholds"])
            y = Yva[c].to_numpy().astype(float)
            for g in GRID:
                p = np.where(pm[c] >= g, DEFAULT_VALUES[c], np.power(10.0, rb[c]))
                a_c = np.where(fixed[c + "_v"], acc_vec(c, p, y), np.nan)
                tot = 0.0
                for c2 in TARGETS:
                    tot += WEIGHTS[c2] * np.nanmean(a_c if c2 == c else fixed[c2])
                tot *= 100.0
                if tot > best:
                    best, th = tot, dict(th, **{c: g})

            keys = list(pw0)
            dls = np.array([pw[k] - pw0[k] for k in keys], dtype=float)
            fin = np.isfinite(dls)
            dep = "   ← 部署现状" if key == "py1_deploy" else ""
            print(f"\n  {c:<4} 加权 {key:<13}  Total {t_base:.3f} → {td:.3f}   Δ {d:+.3f} 分"
                  f"   ({time.time() - t:.0f}s){dep}")
            print(f"       acc {d_base[c]['acc']:.4f}→{dd[c]['acc']:.4f}"
                  f"   real {d_base[c]['real']:.4f}→{dd[c]['real']:.4f}"
                  f"   def {d_base[c]['defv']:.4f}→{dd[c]['defv']:.4f}"
                  f"   Δ {(dd[c]['acc'] - d_base[c]['acc']) * 100 * WEIGHTS[c]:+.4f} 分")
            print(f"       包络（{int(fin.sum())} 井有效）"
                  f"  [{dls[fin].min():+.3f},{dls[fin].max():+.3f}]"
                  f"   正号 {int((dls[fin] > 0).sum())}/{int(fin.sum())}"
                  f"   {'   ← 不跨 0' if fin.all() and dls.min() > 0 else ''}")
            print(f"       重扫最优 {best:.3f}   Δ {best - t_base:+.3f}"
                  f"   （重扫只多 {best - td:+.3f}）   阈值 {th[c]:.3f}")
            rows.append(dict(target=c, key=key, v9=td, d=d, opt=best, d_opt=best - t_base,
                             real=dd[c]["real"], dls=dls, th=th[c]))
            del reg, pred
            gc.collect()

    # ---------- 汇总 ----------
    print("\n" + "=" * 100)
    print("汇总（按 v9 阈值 Δ 排序；看增益是否随「加大 y 权重」单调变化）")
    print("=" * 100)
    print(f"  {'目标':<6}{'加权':<15}{'v9 阈值':>10}{'Δv9':>9}{'重扫最优':>10}{'Δopt':>9}"
          f"{'包络下界':>11}{'包络上界':>11}{'正号':>7}")
    for c in SWEEP_TARGETS:
        for r in sorted([x for x in rows if x["target"] == c], key=lambda x: -x["d"]):
            f = np.isfinite(r["dls"])
            mark = "  ← 部署" if r["key"] == "py1_deploy" else ""
            print(f"  {r['target']:<6}{r['key']:<15}{r['v9']:>10.3f}{r['d']:>+9.3f}"
                  f"{r['opt']:>10.3f}{r['d_opt']:>+9.3f}{r['dls'][f].min():>+11.3f}"
                  f"{r['dls'][f].max():>+11.3f}{int((r['dls'][f] > 0).sum()):>7}{mark}")

    print(f"\n总用时 {time.time() - t0:.0f}s   峰值 {rss()}")


if __name__ == "__main__":
    main()
