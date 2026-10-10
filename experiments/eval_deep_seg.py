"""深分段器整合测试：AUC 有增益能不能变成 Total 增益？

背景
----
deep_seg_probe.py 得到本项目**首次**超过树 AUC 的结果（best-of-14-epoch）：
  POR  深 0.9034 / 均值 0.8968   vs HistGB237 0.8913 / HistGB225 0.8966
  PERM 深 0.9015 / 均值 0.8896   vs HistGB237 0.8846 / HistGB225 0.8906
  SW   深 0.9715 / 均值 0.9628   vs HistGB237 0.9742 / HistGB225 0.9701
+0.012 ~ +0.017 的 AUCΔ 高于历史上所有"换训练方式"的 0.000~0.006 噪声带
（§4.11），所以这是唯一值得继续的深模型轴。

但 AUC 与 Total 的耦合在本项目里**已知是弱的**：
  * §4.11 六种分类器训练方案 AUC 移动 0.000~0.006 → Total 无一为正；
  * §4.14 wellz 让 AUC **下降** 0.005~0.006 → Total **上升** +0.171。
因此 AUC 增益不能推定 Total 增益，必须直接测。

本脚本做什么
------------
1) 复现 v9 基线 73.931（不通过就退出 —— 口径已变的可靠信号）；
2) 把深分段器的 p_def 作为**第三路**加进融合，逐目标扫 (w_deep, 阈值)；
3) 报告 acc / acc_def / acc_real 分解 + 逐目标重扫 Δ（判断是否联合过拟合，
   即 §4.11 的"更差模型 × 更大阈值"指纹）+ 留一井包络（8 井才是实验单元）。

口径纪律
--------
* 纯 numpy + joblib，**不调用任何 sklearn 的 fit，也不 import torch**
  （§5.1：同进程 torch + sklearn fit 在 macOS 上会 SIGSEGV；这里两者都避开）。
* 深 p_def 与缓存的树 reg / 树 p_def / TCN 输出全部是全长 72,779 对齐；
  深概率在标签无效点上是 NaN，评分子集用 valid 掩码，与官方口径一致。
* 三路的顺序无关：只改权重与阈值，不改回归器、不改 blend_w、不重训任何东西。

用法: python experiments/eval_deep_seg.py
"""
from __future__ import annotations

import os
import sys
import gc
import time
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from src.feature import (
    load_wells, make_targets, default_mask, make_features,
    transform_target, inverse_transform_target,
)
from src.model import DEFAULT_CLF_PARAMS, DEFAULT_REG_PARAMS, POR_MONOTONIC
from src.utils import TARGETS, WEIGHTS, EPS, DEFAULT_VALUES
from experiments.eval_v4 import build_tree, tcn_outputs

RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
TCN_TAGS = ["T15_seed42_150ep", "T16_seed43_150ep", "T16_seed44_150ep"]
CFG = {"blend_w": {"POR": 0.50, "PERM": 0.50, "SW": 0.75},
       "thresholds": {"POR": 0.080, "PERM": 0.250, "SW": 0.050},
       "clf_tcn_w": {"POR": 0.60, "PERM": 0.70, "SW": 0.30}}
BASE = 73.931
# 阈值网格：深概率的分布尺度可能与树不同，需要重扫
GRID = np.arange(0.02, 0.71, 0.005)
# 深臂权重网格
WD = np.round(np.arange(0.0, 1.01, 0.10), 3)


def rss() -> str:
    import resource
    return f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576:.0f}MB"


def acc_vec(c: str, p: np.ndarray, t: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    t = np.asarray(t, float)
    if c == "PERM":
        return np.maximum(0.0, 1.0 - np.abs(np.log10(np.maximum(p / np.maximum(t, EPS), EPS))))
    tol = 0.08 if c == "POR" else 0.05
    return np.maximum(0.0, 1.0 - np.abs(p - t) / (tol * (np.abs(t) + EPS)))


def score(accs: dict, Yva: pd.DataFrame, Dva: pd.DataFrame, gid: np.ndarray):
    """逐点等权 Total + 分解 + 逐井 Total。"""
    tot, parts, per = 0.0, {}, {}
    for c in TARGETS:
        a = accs[c]
        y = Yva[c].to_numpy().astype(float)
        valid = np.isfinite(y)
        tot += WEIGHTS[c] * np.nanmean(a)
        d = Dva[c].to_numpy() & valid
        parts[c] = dict(acc=float(np.nanmean(a)),
                        real=float(np.nanmean(a[valid & ~d])) if (valid & ~d).sum() else float("nan"),
                        defv=float(np.nanmean(a[valid & d])) if (valid & d).sum() else float("nan"))
    for g in np.unique(gid):
        m = gid == g
        per[str(g)[:8]] = 100.0 * sum(WEIGHTS[c] * np.nanmean(accs[c][m]) for c in TARGETS)
    return tot * 100, parts, per


def main():
    t0 = time.time()
    deep = np.load(os.path.join(RUNS, "deep_seg_pdef.npz"))
    p_deep = {c: deep[f"deep_{c}"] for c in TARGETS}

    # ---------- 数据 ----------
    wells = load_wells("./train", has_labels=True)
    ids = [w["logId"].iloc[0] for w in wells]
    by = dict(zip(ids, wells))
    tr = pd.concat([by[i] for i in ids[:72]], ignore_index=True)
    va = pd.concat([by[i] for i in ids[72:80]], ignore_index=True)
    va_wells = [by[i] for i in ids[72:80]]
    Ytr, Yva = make_targets(tr), make_targets(va)
    Dva = default_mask(Yva)
    gid = va["logId"].to_numpy()

    # ---------- 树（当前 237 特征，含 wellz）+ TCN ----------
    # 注意：screen_tree_cache 是 wellz 之前的旧缓存（73.727），不可用作基线。
    # 必须走与 tune_wellbore_crosswell.py 相同的新建树路径才能复现 73.931。
    Xva = make_features(va, context=True, priors=True)
    print(f"特征 {Xva.shape[1]} 列   验证 {len(va):,} 点 / 8 井")
    m0, _ = build_tree(tr, weight_tol=True)
    reg_tree, p_tree = {}, {}
    for c in TARGETS:
        reg_tree[c] = np.log10(np.maximum(inverse_transform_target(
            c, pd.Series(m0.regressors[c].predict(Xva), index=Xva.index)).to_numpy(), EPS))
        p_tree[c] = m0.classifiers[c].predict_proba(Xva)[:, 1]

    arts = [joblib.load(os.path.join(RUNS, f"tcn_{t}.joblib")) for t in TCN_TAGS]
    ens, p_tcn = {}, {}
    for art in arts:
        r, p = tcn_outputs(va_wells, art)
        for c in TARGETS:
            ens.setdefault(c, []).append(r[c])
            p_tcn.setdefault(c, []).append(p[c])
    ens = {c: np.mean(v, axis=0) for c, v in ens.items()}
    p_tcn = {c: np.mean(v, axis=0) for c, v in p_tcn.items()}
    del arts
    gc.collect()

    print(f"树 reg/p_def {next(iter(reg_tree.values())).shape[0]:,}   "
          f"TCN {next(iter(p_tcn.values())).shape[0]:,}   深 {next(iter(p_deep.values())).shape[0]:,}")
    for c in TARGETS:
        v = p_deep[c]
        print(f"  {c:<5} 深概率有效点 {int(np.isfinite(v).sum()):,}"
              f"（应为 {int(np.isfinite(Yva[c].to_numpy()).sum()):,}）"
              f"   深概率均值 {np.nanmean(v):.4f}   树 p_def 均值 {p_tree[c].mean():.4f}"
              f"   TCN pdf 均值 {p_tcn[c].mean():.4f}")

    reg_blend = {c: np.where(np.isfinite((1 - CFG["blend_w"][c]) * reg_tree[c]
                                      + CFG["blend_w"][c] * ens[c]),
                             (1 - CFG["blend_w"][c]) * reg_tree[c] + CFG["blend_w"][c] * ens[c], -5.0)
                 for c in TARGETS}

    def pm_of(wd_map):
        """wd_map: {c: 深臂权重}。树臂权重 = (1 - clf_tcn_w - w_deep)。"""
        return {c: (1 - CFG["clf_tcn_w"][c] - wd_map[c]) * p_tree[c]
                         + CFG["clf_tcn_w"][c] * p_tcn[c]
                         + wd_map[c] * p_deep[c] for c in TARGETS}

    def accs_of(pm, ths):
        """NaN 概率（深臂在标签无效点）落到回归分支即可 —— 那些点本来就被 valid 掩掉。"""
        accs = {}
        for c in TARGETS:
            y = Yva[c].to_numpy().astype(float)
            valid = np.isfinite(y)
            pred = np.where(pm[c] >= ths[c], DEFAULT_VALUES[c], np.power(10.0, reg_blend[c]))
            accs[c] = np.where(valid, acc_vec(c, pred, y), np.nan)
        return accs

    # ---------- 基线（w_deep = 0，必须复现 73.931） ----------
    print("\n" + "=" * 96)
    print("基线（v9 配置，深臂权重 0 —— 必须复现 73.931）")
    print("=" * 96)
    pm0 = pm_of({c: 0.0 for c in TARGETS})
    accs0 = accs_of(pm0, CFG["thresholds"])
    t_base, d_base, pw0 = score(accs0, Yva, Dva, gid)
    print(f"  [自检] Total {t_base:.3f}（应复现 {BASE}）"
          f"   {'✓ 复现成功' if abs(t_base - BASE) <= 0.02 else '✗ 未能复现'}")
    if abs(t_base - BASE) > 0.02:
        print("  ✗ 基线未复现，退出（缓存或口径已变）")
        return
    print("  [自检] 分解 " + "  ".join(
        f"{c} acc {d_base[c]['acc']:.4f}/real {d_base[c]['real']:.4f}/def {d_base[c]['defv']:.4f}"
        for c in TARGETS))
    print(f"  逐井 " + "  ".join(f"{k[:6]} {v:.3f}" for k, v in pw0.items()))
    print(f"  {rss()}")

    # ---------- 逐目标：扫 (w_deep, 阈值) ----------
    # 先在目标内自由扫，得到每目标最优 (w_deep, 阈值)；再汇总成 Total。
    # 注意：三个目标在官方口径下是逐点等权、互相独立的，所以逐目标最优即联合最优。
    print("\n" + "=" * 96)
    print("逐目标扫描：w_deep ∈ [0,1] 步 0.10 × 阈值 ∈ [0.02,0.70] 步 0.005")
    print("=" * 96)
    chosen = {c: (0.0, CFG["thresholds"][c]) for c in TARGETS}
    best_t = {c: 0.0 for c in TARGETS}
    tab = []
    for c in TARGETS:
        cw = CFG["clf_tcn_w"][c]
        for wd in WD:
            w_tree = 1.0 - cw - wd
            if w_tree < -1e-9:
                continue
            pc = w_tree * p_tree[c] + cw * p_tcn[c] + wd * p_deep[c]
            pc = np.where(np.isfinite(pc), pc, np.nan)
            y = Yva[c].to_numpy().astype(float)
            valid = np.isfinite(y)
            a_arr = None
            bt, btv = -1.0, CFG["thresholds"][c]
            for g in GRID:
                pred = np.where(pc >= g, DEFAULT_VALUES[c], np.power(10.0, reg_blend[c]))
                a = np.where(valid, acc_vec(c, pred, y), np.nan)
                v = float(WEIGHTS[c] * np.nanmean(a))
                if v > bt:
                    bt, btv, a_arr = v, g, a
            tab.append((c, wd, btv, bt * 100, a_arr))
    print(f"  {'目标':<6}{'w_deep':>8}{'阈值':>9}{'贡献':>10}   说明")
    rows = []
    for c in TARGETS:
        cands = [r for r in tab if r[0] == c]
        b = max(cands, key=lambda r: r[3])
        c0 = next(r for r in cands if abs(r[1] - 0.0) < 1e-9)
        print(f"  {c:<6}{b[1]:>8.2f}{b[2]:>9.3f}{b[3]:>10.3f}"
              f"   w_deep=0 时 {c0[3]:.3f}（贡献 Δ {(b[3] - c0[3]):+.3f} 分）"
              f"   {'← 深臂被选中' if b[1] > 1e-9 else ''}")
        chosen[c] = (float(b[1]), float(b[2]))
        best_t[c] = b[3]
        rows.append(b)

    # 汇总成 Total（逐目标独立，直接相加）
    accs_best = {c: next(r[4] for r in tab if r[0] == c and abs(r[1] - chosen[c][0]) < 1e-9
                         and abs(r[2] - chosen[c][1]) < 1e-9) for c in TARGETS}
    t_deep, d_deep, pw_deep = score(accs_best, Yva, Dva, gid)
    d_tot = t_deep - BASE

    print("\n" + "=" * 96)
    print(f"汇总（每目标取自己的最优 (w_deep, 阈值)）   Total {BASE} → {t_deep:.3f}   Δ {d_tot:+.3f} 分")
    print("=" * 96)
    for c in TARGETS:
        b, dd = d_base[c], d_deep[c]
        print(f"  {c:<5} w_deep {chosen[c][0]:.2f}  阈值 {CFG['thresholds'][c]:.3f}→{chosen[c][1]:.3f}"
              f"   acc {b['acc']:.4f}→{dd['acc']:.4f}   real {b['real']:.4f}→{dd['real']:.4f}"
              f"   def {b['defv']:.4f}→{dd['defv']:.4f}"
              f"   Δ {(dd['acc'] - b['acc']) * 100 * WEIGHTS[c]:+.4f} 分")

    # ---------- 留一井包络（8 井才是实验单元） ----------
    print("\n留一井 Δ（逐井 Total，每井内逐点等权）")
    keys = list(pw0)
    dls = np.array([pw_deep[k] - pw0[k] for k in keys], dtype=float)
    for k, x in zip(keys, dls):
        print(f"  {k}  {pw0[k]:.3f} → {pw_deep[k]:.3f}   Δ {x:+.3f}")
    fin = np.isfinite(dls)
    print(f"  包络（{int(fin.sum())} 井有效）"
          f"  [{'%+.3f' % dls[fin].min() if fin.any() else 'na'},"
          f"{'%+.3f' % dls[fin].max() if fin.any() else 'na'}]"
          f"   正号 {int((dls[fin] > 0).sum())}/{int(fin.sum())}"
          f"   {'   ← 包络不跨 0' if fin.all() and dls.min() > 0 else ''}")

    print(f"\n总用时 {time.time() - t0:.0f}s   峰值 {rss()}")


if __name__ == "__main__":
    main()
