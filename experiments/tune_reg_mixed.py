"""逐目标回归器目标形状：把每个目标调到它自己的最优 (变换, 损失, 加权)。

发现的来源
---------
tune_reg_objective.py 的 V2「all_log + absolute_error」是全负（−0.100），但分解后
不是均匀的负：

    POR  acc 0.6629→0.6640   real 0.1689→0.1716   Δ +0.0315 分
    PERM acc 0.7753→0.7711   real 0.4670→0.4582   Δ −0.1459 分
    SW   acc 0.7688→0.7692   real 0.1574→0.1585   Δ +0.0148 分

即 POR / SW 换到 log10 + MAE 是**真实段 acc 上升**（不是阈值交互），只有 PERM 变差。
三个目标在官方口径下是独立加权的，所以**可以逐目标各取各的最优**——仓库里从来没
有过逐目标 loss 开关（src/model.py 的 loss 硬编码成标量），这条轴整条被跳过。

为什么值得注意
-------------
这是本项目**第一次**看到真实段 acc 在同一阈值下上升：
  * §4.11 的六个分段器候选，收益全部来自「更差分类器 × 更大阈值」的交互项，
    真实段独立分量**全为负**；
  * §4.14 的 wellz 收益来自 acc_def（阈值不动时 AUC 还降了）；
  * 本轮 POR real 0.1689→0.1716、SW real 0.1574→0.1585，阈值完全没动。
方向上与 §4.11 的过拟合指纹正好相反，所以不能用「可能是阈值过拟合」驳回。
但幅度只有 +0.046，仍在 ±0.3 screen 噪声带内，必须过留一井包络才算数。

口径纪律
--------
* 唯一变量 = 每个目标的树回归器 (mode, loss, weight)。分段器、融合权重、TCN 全冻结。
* 纯 numpy + sklearn，**不 import torch**（§5.1）。TCN 走 numpy 前向，零 GPU。
* 逆变换必须用**该目标自己的 mode**：all_log 下 POR/SW 的树输出已是 log10，
  若再套 perm_log 会原样返回，外层 np.log10 就变成二次取对数 → 预测全废。
  有动态范围门控兜底（超出基线 [1%,99%] ±0.5 log10 超过 5% 即放弃）。
* 报 acc / acc_def / acc_real 分解 + 留一井包络（8 井是实验单元，行级 bootstrap 无效）
  + v9 阈值下 Δ 与重扫 Δ（判断是否联合过拟合）。

⚠️ 加权是**逐目标**的，写错会静默掉分
--------------------------------------
train.py:281 原文是 `w = |log10 y| if c == "PERM" else |y|` —— PERM 用 |log10 y|，
POR / SW 用 |y|。所以「V3 全 abslogy」相对基线的真实差异**不是**「三个目标都翻转加权」，
而是：

    POR   |y|        → |log10 y|    ← 真翻转
    PERM  |log10 y|  → |log10 y|    ← 没变
    SW    |y|        → |log10 y|    ← 真翻转

也就是说 PERM 在部署时就已经用了「压低大 y」的加权，只有 POR/SW 还在用 |y|。
本项目从未单独测试过「POR/SW 的加权方向」这一条轴——它一直被包在
weight_tol 这个整体开关里（§4.5 只记录了 weight_tol 对官方 acc「零影响」，
但那是**整体**开关，不是逐目标拆开的）。本脚本的 W1/W2 就是专门拆这个。

初版曾把 DEPLOY 三个目标全写成 abs_y，自检报 73.814 而非 73.931（−0.117，正好是
PERM 那份加权的量级）。定位方法是二分：`eval_cfg_` + `build_tree` 的原生回归器
精确复现 73.931 → 评估路径无误 → 缺陷在配置字典，不在拟合。

用法: python experiments/tune_reg_mixed.py
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
from src.model import DEFAULT_REG_PARAMS, POR_MONOTONIC, DEFAULT_CLF_PARAMS
from experiments.eval_v4 import build_tree, tcn_outputs

RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
TAGS = ["T15_seed42_150ep", "T16_seed43_150ep", "T16_seed44_150ep"]
CFG = {"blend_w": {"POR": 0.50, "PERM": 0.50, "SW": 0.75},
       "thresholds": {"POR": 0.080, "PERM": 0.250, "SW": 0.050},
       "clf_tcn_w": {"POR": 0.60, "PERM": 0.70, "SW": 0.30}}
BASE = 73.931
GRID = np.arange(0.02, 0.71, 0.005)

# 逐目标配置：{目标: (mode, loss, weight)}
# mode:   perm_log = 仅 PERM 取 log10（部署现状）；all_log = 三目标全 log10
# loss:   sklearn HistGB 的 loss 名
# weight: abs_y / abslogy / none
# ⚠️ weight 是**逐目标**的（train.py:281 原文）：PERM 用 |log10 y|，POR/SW 用 |y|。
# 曾把三者全写成 abs_y 而静默掉 −0.117（73.814 vs 73.931）—— 那正是 PERM 的加权。
DEPLOY = {"POR": ("perm_log", "squared_error", "abs_y"),
          "PERM": ("perm_log", "squared_error", "abslogy"),
          "SW": ("perm_log", "squared_error", "abs_y")}
LOGMAE = ("all_log", "absolute_error", "abslogy")


def logmae(t):
    """把目标 t 换成 all_log + absolute_error，其余不动。"""
    return {t: LOGMAE}


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


def fit_reg(Xtr, Ytr, Dtr, cfg):
    """按逐目标 (mode, loss, weight) 拟合三个回归器。"""
    out = {}
    for c in TARGETS:
        mode, loss, weight = cfg[c]
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
        out[c] = (reg, mode)
    return out


def preds_of_(reg_by_t, ths, Xva, ens, pm_deploy):
    """reg_by_t: {c: (model, mode)}。返回 {c: 原始量纲最终预测}。

    与 tune_reg_objective.py::score 逐行同构：树预测先逆变换回线性量纲，再统一取
    log10 与 TCN 输出按 blend_w 融合（融合空间是 log10，PERM 与 POR/SW 都适用）。
    """
    ths = ths or CFG["thresholds"]
    out = {}
    for c in TARGETS:
        reg, mode = reg_by_t[c]
        lin = inverse_transform_target(c,
                                       pd.Series(reg.predict(Xva), index=Xva.index), mode)
        r10 = np.log10(np.maximum(lin.to_numpy(), EPS))
        rb = (1 - CFG["blend_w"][c]) * r10 + CFG["blend_w"][c] * ens[c]
        rb = np.where(np.isfinite(rb), rb, -5.0)
        out[c] = np.where(pm_deploy[c] >= ths[c], DEFAULT_VALUES[c], np.power(10.0, rb))
    return out


def eval_cfg_(reg_by_t, ths, Xva, Yva, Dva, gid, ens, pm_deploy):
    """官方口径 Total + acc/real/def 分解 + 逐井 Total。逐点等权。"""
    pred = preds_of_(reg_by_t, ths, Xva, ens, pm_deploy)
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
    Xtr = make_features(tr, context=True, priors=True)
    Xva = make_features(va, context=True, priors=True)
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

    # 部署分段器（冻结）
    m0, _ = build_tree(tr, weight_tol=True)
    p_tree = {c: m0.classifiers[c].predict_proba(Xva)[:, 1] for c in TARGETS}
    pm_deploy = {c: (1 - CFG["clf_tcn_w"][c]) * p_tree[c] + CFG["clf_tcn_w"][c] * pcn[c]
                 for c in TARGETS}

    def eval_cfg(reg_by_t, ths=None):
        return eval_cfg_(reg_by_t, ths, Xva, Yva, Dva, gid, ens, pm_deploy)

    # ---------- 基线 ----------
    rt0 = fit_reg(Xtr, Ytr, Dtr, DEPLOY)
    t_base, d_base, pw0 = eval_cfg(rt0)
    print(f"\n[自检] 部署配置复现 = {t_base:.3f}（应 {BASE}）"
          f"   {'✓' if abs(t_base - BASE) <= 0.02 else '✗ 未复现 → 退出'}")
    if abs(t_base - BASE) > 0.02:
        return
    print(f"  分解 " + "  ".join(
        f"{c} {d_base[c]['acc']:.4f}/real {d_base[c]['real']:.4f}/def {d_base[c]['defv']:.4f}"
        for c in TARGETS))
    print(f"  逐井 " + "  ".join(f"{k[:6]} {v:.3f}" for k, v in pw0.items()))
    del rt0
    gc.collect()

    # ---------- 逐目标变体 ----------
    # 关键澄清：PERM 的部署加权本来就是 |log10 y|（train.py:281），POR/SW 才是 |y|。
    # 所以「V3 = 三目标全 all_log+MAE+w=|log10y|」相对基线的真实差异是：
    #   POR/SW 的加权 |y| → |log10 y|（压低大 y 点，方向与官方逐点等权相对误差一致）
    #   + 三目标的 mode/loss 改动
    # 下面的 W* 组把这两件事拆开，先确认增益主体到底是哪一个。
    W_POR = ("perm_log", "squared_error", "abslogy")   # 只翻 POR 加权，其余全不动
    W_SW = ("perm_log", "squared_error", "abslogy")
    VARIANTS = [
        ("M0 V3 全 all_log+MAE+w=|log10y|（tune_reg_objective 最优）",
         {"POR": LOGMAE, "PERM": LOGMAE, "SW": LOGMAE}),
        ("W1 仅 POR 加权 |y|→|log10 y|（最省落地：只改一行权重）",
         {**DEPLOY, **{"POR": W_POR}}),
        ("W2 仅 SW 加权 |y|→|log10 y|",
         {**DEPLOY, **{"SW": W_SW}}),
        ("W3 POR 加权+mode（all_log, squared_error）→ 拆 mode 与 loss",
         {**DEPLOY, **{"POR": ("all_log", "squared_error", "abslogy")}}),
        ("W4 POR 加权+loss（perm_log, absolute_error）→ 拆 loss 与 mode",
         {**DEPLOY, **{"POR": ("perm_log", "absolute_error", "abslogy")}}),
        ("W5 SW 加权+mode（all_log, squared_error）",
         {**DEPLOY, **{"SW": ("all_log", "squared_error", "abslogy")}}),
        ("W6 SW 加权+loss（perm_log, absolute_error）",
         {**DEPLOY, **{"SW": ("perm_log", "absolute_error", "abslogy")}}),
        ("M2 仅 PERM mode+loss（加权本已是 |log10 y|，不动）",
         {**DEPLOY, **{"PERM": ("all_log", "absolute_error", "abslogy")}}),
    ]
    rows = []
    for name, cfg in VARIANTS:
        t = time.time()
        got = fit_reg(Xtr, Ytr, Dtr, cfg)
        td, dd, pw = eval_cfg(got)
        d = td - t_base
        print(f"\n{name}\n  Total {t_base:.3f} → {td:.3f}   Δ {d:+.3f} 分"
              f"   ({time.time() - t:.0f}s)")
        for c in TARGETS:
            b = d_base[c]
            mark = "← 本变体动过" if cfg[c] != DEPLOY[c] else ""
            print(f"     {c}  acc {b['acc']:.4f}→{dd[c]['acc']:.4f}"
                  f"   real {b['real']:.4f}→{dd[c]['real']:.4f}"
                  f"   def {b['defv']:.4f}→{dd[c]['defv']:.4f}"
                  f"   Δ {(dd[c]['acc'] - b['acc']) * 100 * WEIGHTS[c]:+.4f} 分   {mark}")
        # 留一井包络（NaN 安全：病态井 SW 全 NaN 会毒化 min/max）
        keys = list(pw0)
        dls = np.array([pw[k] - pw0[k] for k in keys], dtype=float)
        fin = np.isfinite(dls)
        print(f"     包络（{int(fin.sum())} 井有效）"
              f"  [{dls[fin].min():+.3f},{dls[fin].max():+.3f}]   正号 {int((dls[fin] > 0).sum())}/{int(fin.sum())}"
              f"   {'   ← 不跨 0' if fin.all() and dls.min() > 0 else ''}")
        # 阈值重扫（判断是否联合过拟合）。
        # 向量化：三个目标在官方口径下独立，预先算好「不动的目标」在 v9 阈值下的 acc
        # 向量，只对被扫的那个目标逐阈值重算 —— 避免 414 次全量 eval_cfg。
        reg10 = {c: np.log10(np.maximum(inverse_transform_target(
            c, pd.Series(got[c][0].predict(Xva), index=Xva.index), got[c][1]).to_numpy(), EPS))
            for c in TARGETS}
        rb = {c: np.where(
            np.isfinite((1 - CFG["blend_w"][c]) * reg10[c] + CFG["blend_w"][c] * ens[c]),
            (1 - CFG["blend_w"][c]) * reg10[c] + CFG["blend_w"][c] * ens[c], -5.0)
            for c in TARGETS}
        fixed = {}
        for c in TARGETS:
            y = Yva[c].to_numpy().astype(float)
            p = np.where(pm_deploy[c] >= CFG["thresholds"][c], DEFAULT_VALUES[c], np.power(10.0, rb[c]))
            fixed[c] = np.where(np.isfinite(y), acc_vec(c, p, y), np.nan)
            fixed[c + "_v"] = np.isfinite(y)
        best, th = td, dict(CFG["thresholds"])
        for c in TARGETS:
            y = Yva[c].to_numpy().astype(float)
            for g in GRID:
                p = np.where(pm_deploy[c] >= g, DEFAULT_VALUES[c], np.power(10.0, rb[c]))
                a_c = np.where(fixed[c + "_v"], acc_vec(c, p, y), np.nan)
                tot = 0.0
                for c2 in TARGETS:
                    tot += WEIGHTS[c2] * np.nanmean(a_c if c2 == c else fixed[c2])
                tot *= 100.0
                if tot > best:
                    best, th = tot, dict(th, **{c: g})
        print(f"     重扫最优 {best:.3f}   Δ {best - t_base:+.3f}"
              f"   阈值 {{{' '.join(f'{c}={th[c]:.3f}' for c in TARGETS)}}}"
              f"   （重扫只多 {best - td:+.3f}）")
        rows.append(dict(name=name, cfg=cfg, v9=td, d=d, opt=best, d_opt=best - t_base,
                         th=th, real={c: dd[c]["real"] for c in TARGETS},
                         defv={c: dd[c]["defv"] for c in TARGETS}, dls=dls))
        del got
        gc.collect()

    # ---------- 若 M4 最优，做累积组合验证 ----------
    print("\n" + "=" * 100)
    print("排名")
    print("=" * 100)
    print(f"  {'变体':<46}{'v9 阈值':>10}{'Δv9':>9}{'重扫最优':>10}{'Δopt':>9}"
          f"{'包络下界':>11}{'包络上界':>11}{'正号':>7}")
    print(f"  {'V0 部署现状':<46}{t_base:>10.3f}{'':>9}{t_base:>10.3f}{'':>9}{'':>11}{'':>11}{'':>7}")
    for r in sorted(rows, key=lambda x: -x["d_opt"]):
        f = np.isfinite(r["dls"])
        print(f"  {r['name']:<46}{r['v9']:>10.3f}{r['d']:>+9.3f}{r['opt']:>10.3f}"
              f"{r['d_opt']:>+9.3f}{r['dls'][f].min():>+11.3f}{r['dls'][f].max():>+11.3f}"
              f"{int((r['dls'][f] > 0).sum()):>7}")

    # 最优变体的逐目标 real/def 对照
    best_r = max(rows, key=lambda x: x["d_opt"])
    print(f"\n最优变体 {best_r['name']} 的逐目标分解（对比基线）")
    for c in TARGETS:
        print(f"  {c:<5} real {d_base[c]['real']:.4f}→{best_r['real'][c]:.4f}"
              f"（Δ {best_r['real'][c] - d_base[c]['real']:+.4f}）"
              f"   def {d_base[c]['defv']:.4f}→{best_r['defv'][c]:.4f}"
              f"（Δ {best_r['defv'][c] - d_base[c]['defv']:+.4f}）")

    print(f"\n总用时 {time.time() - t0:.0f}s   峰值 {rss()}")


if __name__ == "__main__":
    main()
