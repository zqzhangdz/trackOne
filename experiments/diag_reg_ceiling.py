"""回归器轴的上界估计：真值分段下，换损失函数还能拿多少？

为什么这个之前没人做
-------------------
§4.15 的预算表把可追分数全部记在**分段器**这一步：
    v9 实际 73.931 → 真值分段上界 76.336 → 剩余 +2.405，全部在分段器。
但那个 76.336 的算法是「真值掩码 + **现有回归器**」。也就是说：
    * 它假设分段器是神（拿不到，AUC 上限 0.90）；
    * 同时把回归器**冻结**在部署配置上。
所以 +2.405 只是分段器轴的预算，**回归器轴从来没有任何预算记录**。

官方口径折算到 log10 空间后三目标都是 |log10(ŷ/y)|，带 POR 0.03342 / PERM 1.0 /
SW 0.02119。而实测真实段的一阶 log10 MAE 是 POR 0.748（22× 带）、PERM 1.043（1.04×）、
SW 0.0430（2.03×）。PERM 与 SW 的一阶误差只比带宽一点 —— 这是**唯一**一阶误差接近带
的目标，也就是唯一「换损失形状可能咬得住」的地方。部署现状是：
    PERM 在 log10 空间用 squared_error（对数均方，不是官方的对数绝对）；
    POR/SW 在**线性空间**用 squared_error（既非对数也非绝对）；
    再叠 weight_tol 的 w=|y|，把大 y 的点权重抬高，方向与官方「逐点等权相对误差」相反。
这三个地方**从来没有任何代码路径能改**（src/model.py 的 loss 是硬编码，且 p.update
之后再传 loss 会 TypeError），所以整条轴在旧口径时代就被"试不了"这个理由跳过了。

本脚本做什么
------------
在**真值分段**（oracle 掩码，无分类器噪声）下比较四种回归器目标。因为分段器被换成神，
任何分数差异都 100% 归因于回归器 —— 这是隔离回归器轴的唯一办法。

  A perm_log + squared_error + w=|y|      部署现状
  B perm_log + absolute_error  + w=|y|    log10 空间直接优化官方形状
  C all_log  + absolute_error  + w=|y|    POR/SW 也进 log10
  D all_log  + absolute_error  + w=|log10 y|  加权与目标量纲一致

同时报告「现状分段器 + 换回归器」（73.931 基线可比）与「真值分段 + 换回归器」两个口径，
前者是可部署的真实收益，后者是该轴的绝对上界。

口径纪律
--------
* 纯 numpy + sklearn，**不 import torch**（§5.1）。TCN 只走 numpy 前向，零 GPU。
* 唯一变量 = 树回归器的 (目标变换, 损失, 加权)。分段器、融合权重、TCN 全冻结。
* 官方口径逐点等权；报 acc / acc_def / acc_real 分解；8 井是实验单元。

用法: python experiments/diag_reg_ceiling.py
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
ORACLE_BASE = 76.336          # 真值分段 + 部署回归器（§4.2 / §4.15）
GRID = np.arange(0.02, 0.71, 0.005)


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


def fit_reg(Xtr: pd.DataFrame, Ytr: pd.DataFrame, Dtr: pd.DataFrame,
            mode: str, loss: str, weight: str) -> dict:
    """只拟合三个回归器（不动分类器）。mode 同时决定正向与逆变换，必须成对。"""
    out = {}
    for c in TARGETS:
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
        yt = transform_target(c, y.loc[real], mode)
        reg.fit(Xv.loc[real], yt)
        if weight != "none":
            yv = y.to_numpy()[real]
            w = (np.abs(np.log10(np.maximum(yv, EPS))) if weight == "abslogy"
                 else np.abs(yv))
            w = np.clip(w, np.percentile(w, 5), np.percentile(w, 95))
            w = w / w.mean()
            reg.fit(Xv.loc[real], yt, sample_weight=w)
        out[c] = reg
    return out


def score(pred, Yva, Dva, gid):
    """pred: {c: 原始量纲的最终预测}（判默认→DEFAULT_VALUES，否则→回归值）。

    调用方必须传**原始量纲**预测，不要传已算好的 acc 向量（会被二次套 acc_vec）。
    """
    tot, parts, per, accs = 0.0, {}, {}, {}
    for c in TARGETS:
        y = Yva[c].to_numpy().astype(float)
        valid = np.isfinite(y)
        d = Dva[c].to_numpy() & valid
        a = np.where(valid, acc_vec(c, pred[c], y), np.nan)
        accs[c] = a
        tot += WEIGHTS[c] * np.nanmean(a)
        parts[c] = dict(acc=float(np.nanmean(a)),
                        real=float(np.nanmean(a[valid & ~d])) if (valid & ~d).sum() else float("nan"),
                        defv=float(np.nanmean(a[valid & d])) if (valid & d).sum() else float("nan"))
    for g in np.unique(gid):
        m = gid == g
        per[str(g)[:8]] = 100.0 * sum(WEIGHTS[c] * np.nanmean(accs[c][m]) for c in TARGETS)
    return tot * 100, parts, per


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

    # TCN（只 numpy 前向，纯读缓存 joblib）
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

    # 部署分段器（用于「现状分段器 + 换回归器」口径）
    m0, _ = build_tree(tr, weight_tol=True)
    p_tree = {c: m0.classifiers[c].predict_proba(Xva)[:, 1] for c in TARGETS}
    pt0 = {c: np.log10(np.maximum(inverse_transform_target(
        c, pd.Series(m0.regressors[c].predict(Xva), index=Xva.index)).to_numpy(), EPS))
        for c in TARGETS}

    def reg_blend_of(rt):
        return {c: np.where(
            np.isfinite((1 - CFG["blend_w"][c]) * rt[c] + CFG["blend_w"][c] * ens[c]),
            (1 - CFG["blend_w"][c]) * rt[c] + CFG["blend_w"][c] * ens[c], -5.0) for c in TARGETS}

    def pred_of(rt, oracle):
        """rt: {c: log10 树预测}。oracle=True 用真值掩码，False 用部署分段器。"""
        rb = reg_blend_of(rt)
        out = {}
        for c in TARGETS:
            if oracle:
                mask = Dva[c].to_numpy()
            else:
                cw = CFG["clf_tcn_w"][c]
                pm = (1 - cw) * p_tree[c] + cw * pcn[c]
                mask = pm >= CFG["thresholds"][c]
            out[c] = np.where(mask, DEFAULT_VALUES[c], np.power(10.0, rb[c]))
        return out

    def run(rt, oracle, tag):
        pred = pred_of(rt, oracle)
        tot, parts, per = score(pred, Yva, Dva, gid)
        return tot, parts, per

    # ---------- 两个基线 ----------
    rt0 = pt0
    t_base, d_base, pw0 = run(rt0, False, "V0 deployed")
    t_orac0, d_orac0, pw_o0 = run(rt0, True, "V0 oracle")
    print(f"\n[自检] 现状分段器 + 部署回归器 = {t_base:.3f}（应 {BASE}）"
          f"   {'✓' if abs(t_base - BASE) <= 0.02 else '✗ 未复现 → 退出'}")
    if abs(t_base - BASE) > 0.02:
        return
    print(f"[自检] 真值分段 + 部署回归器   = {t_orac0:.3f}（记录值 {ORACLE_BASE}）"
          f"   Δ {t_orac0 - ORACLE_BASE:+.3f}（分段器轴上界）")
    print(f"  分解 " + "  ".join(
        f"{c} {d_base[c]['acc']:.4f}/real {d_base[c]['real']:.4f}/def {d_base[c]['defv']:.4f}"
        for c in TARGETS))
    print(f"  oracle 分解 " + "  ".join(
        f"{c} {d_orac0[c]['acc']:.4f}/real {d_orac0[c]['real']:.4f}/def {d_orac0[c]['defv']:.4f}"
        for c in TARGETS))

    # ---------- 变体 ----------
    VARIANTS = [
        ("A perm_log + squared_error + w=|y|  (部署现状)", "perm_log", "squared_error", "abs_y"),
        ("B perm_log + absolute_error  + w=|y|  (log10 MAE=官方形状)", "perm_log", "absolute_error", "abs_y"),
        ("C all_log  + absolute_error  + w=|y|  (POR/SW 进 log10)", "all_log", "absolute_error", "abs_y"),
        ("D all_log  + absolute_error  + w=|log10 y|", "all_log", "absolute_error", "abslogy"),
        ("E perm_log + squared_error  无 weight_tol", "perm_log", "squared_error", "none"),
    ]
    rows = []
    for name, mode, loss, weight in VARIANTS:
        t = time.time()
        got = fit_reg(Xtr, Ytr, Dtr, mode, loss, weight)
        # 逆变换必须用本变体的 mode（否则 all_log 下会二次取对数 → 预测全废）
        rt = {c: np.log10(np.maximum(inverse_transform_target(
            c, pd.Series(got[c].predict(Xva), index=Xva.index), mode).to_numpy(), EPS))
            for c in TARGETS}
        # 动态范围健全性：超出基线 [1%,99%] ±0.5 log10 超过 5% 即逆变换不匹配
        bad = False
        for c in TARGETS:
            lo, hi = float(np.percentile(pt0[c], 1)), float(np.percentile(pt0[c], 99))
            f = float(np.mean((rt[c] < lo - 0.5) | (rt[c] > hi + 0.5)))
            if f > 0.05:
                print(f"\n{name}\n  ✗ {c} 预测 {f * 100:.1f}% 超出动态范围 "
                      f"[{lo:.2f},{hi:.2f}] → 逆变换 mode 不匹配，放弃")
                bad = True
        if bad:
            del got
            gc.collect()
            continue

        td, dd, pw = run(rt, False, name)
        to, do, _ = run(rt, True, name)
        d = td - t_base
        print(f"\n{name}\n  现状分段器 {t_base:.3f} → {td:.3f}   Δ {d:+.3f}"
              f"   真值分段 {t_orac0:.3f} → {to:.3f}   Δ {to - t_orac0:+.3f}"
              f"   (回归器轴贡献)   ({time.time() - t:.0f}s)")
        for c in TARGETS:
            b = d_base[c]
            print(f"     {c}  acc {b['acc']:.4f}→{dd[c]['acc']:.4f}"
                  f"   real {b['real']:.4f}→{dd[c]['real']:.4f}"
                  f"   Δ {(dd[c]['acc'] - b['acc']) * 100 * WEIGHTS[c]:+.4f} 分"
                  f"   |  oracle real {do[c]['real']:.4f}"
                  f"（Δ {do[c]['real'] - d_orac0[c]['real']:+.4f}）")
        rows.append(dict(name=name, deployed=td, d_dep=d, oracle=to, d_orac=to - t_orac0,
                         rt=rt, pw=pw))
        del got
        gc.collect()

    # ---------- 阈值重扫：排除「更差回归器 × 更大阈值」的联合过拟合（§4.11 指纹） ----------
    print("\n" + "=" * 96)
    print("阈值重扫（回归器最优，分段器仍用现状；只看 Δ 是否仍为正）")
    print("=" * 96)
    for r in rows:
        th = dict(CFG["thresholds"])
        best = r["deployed"]
        for c in TARGETS:
            rb = reg_blend_of(r["rt"])
            for g in GRID:
                out = {}
                for c2 in TARGETS:
                    cw = CFG["clf_tcn_w"][c2]
                    pm = (1 - cw) * p_tree[c2] + cw * pcn[c2]
                    mask = pm >= (g if c2 == c else th[c2])
                    out[c2] = np.where(mask, DEFAULT_VALUES[c2], np.power(10.0, rb[c2]))
                tot, _, _ = score(out, Yva, Dva, gid)
                if tot > best:
                    best, th = tot, dict(th, **{c: g})
        r["opt"] = best
        print(f"  {r['name']:<56} v9 阈值 {r['deployed']:.3f} (Δ{r['d_dep']:+.3f})"
              f"  重扫 {best:.3f} (Δ{best - t_base:+.3f})"
              f"  重扫只多 {best - r['deployed']:+.3f}  阈值 {{{' '.join(f'{c}={th[c]:.3f}' for c in TARGETS)}}}")

    print("\n" + "=" * 96)
    print("回归器轴总结（真值分段下隔离，§4.15 从未预算过这一轴）")
    print("=" * 96)
    print(f"  {'变体':<56}{'现状分段器':>12}{'Δ':>9}{'真值分段':>12}{'Δ(轴上界)':>12}")
    print(f"  {'V0 部署现状':<56}{t_base:>12.3f}{'':>9}{t_orac0:>12.3f}{'':>12}")
    for r in sorted(rows, key=lambda x: -x["d_orac"]):
        print(f"  {r['name']:<56}{r['deployed']:>12.3f}{r['d_dep']:>+9.3f}"
              f"{r['oracle']:>12.3f}{r['d_orac']:>+12.3f}")

    print(f"\n总用时 {time.time() - t0:.0f}s   峰值 {rss()}")


if __name__ == "__main__":
    main()
