"""生死测试：逐井阈值标定（transductive 校准）里到底有没有分。

动机
----
分类器在全局阈值下**系统性把真实段误判成默认段**（§4.2：POR 判默认 0.7536 vs 真值 0.6003，
PERM 0.7781 vs 0.6063）。而**默认段占比是井级常数，且井间差异巨大** —— screen 后 8 井的
POR 占比 60.0% 对比训练集 66.6%，其中病态井 c7611b0148bb4b 的 POR 占比只有 0.001。
**一个全局阈值不可能同时服务占比 0.001 与 0.95 的两口井。**

本脚本先做最便宜的生死信号：如果每口井的占比已知（oracle），逐井阈值能拿回多少分？
  * 拿不回 → 方向当场死，别再建估计器。
  * 拿得回 → 剩下才是"能否从特征无监督地估计井级占比"，那是另一件事。

只用缓存前向，**不 fit 任何模型、不 import torch**（遵守 §5.1 同进程运行时约束）。
用法: python experiments/tune_transductive_cal.py
"""
from __future__ import annotations
import os, sys, gc, time, warnings
warnings.filterwarnings("ignore")
import joblib, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.feature import load_wells, make_targets, default_mask
from src.utils import DEFAULT_VALUES, EPS, TARGETS, WEIGHTS

RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
T108 = {"T15": "T15_seed42_150ep", "T16a": "T16_seed43_150ep", "T16b": "T16_seed44_150ep"}
# v7 配置（±10/w21，225 特征）—— 缓存就是这个配置下的前向
V7 = {"blend_w": {"POR": 0.50, "PERM": 0.50, "SW": 0.75},
      "thresholds": {"POR": 0.130, "PERM": 0.270, "SW": 0.050},
      "clf_tcn_w": {"POR": 0.60, "PERM": 0.70, "SW": 0.30}}


def acc_vec(c: str, pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """官方口径逐点准确率。PERM 恒 1.0 decade 比值容差，无系数无分母。"""
    p = np.asarray(pred, float); t = np.asarray(true, float)
    if c == "PERM":
        return np.maximum(0.0, 1.0 - np.abs(np.log10(np.maximum(p / np.maximum(t, EPS), EPS))))
    tol = 0.08 if c == "POR" else 0.05
    return np.maximum(0.0, 1.0 - np.abs(p - t) / (tol * (np.abs(t) + EPS)))


def main() -> None:
    t0 = time.time()
    wells = load_wells("./train", has_labels=True)
    ids = [w["logId"].iloc[0] for w in wells]
    val_ids = ids[72:]
    va = pd.concat([w for w in wells if w["logId"].iloc[0] in set(val_ids)], ignore_index=True)
    del wells; gc.collect()
    va = va.sort_index().reset_index(drop=True)
    Y = make_targets(va)
    DM = default_mask(Y)
    well = va["logId"].to_numpy()

    cache = joblib.load(os.path.join(RUNS, "screen_tree_cache.joblib"))
    rt, pt = cache["reg"], cache["p_def"]
    store = {k: v for k, v in np.load(os.path.join(RUNS, "tcn_screen_outputs.npz")).items()}
    assert len(va) == 72779, f"验证集应为 72779 行，实得 {len(va)}"

    # 每目标：融合概率 pm、融合回归（log10 空间）→ 反变换回原空间
    pm, pred_real, y_true, is_def_true = {}, {}, {}, {}
    for c in TARGETS:
        w, cw = V7["blend_w"][c], V7["clf_tcn_w"][c]
        ens = np.mean([store[f"{T108[t]}__{c}__reg"] for t in T108], axis=0)
        pcn = np.mean([store[f"{T108[t]}__{c}__pdf"] for t in T108], axis=0)
        pm[c] = (1 - cw) * pt[c] + cw * pcn
        reg = (1 - w) * rt[c] + w * ens
        reg = np.where(np.isfinite(reg), reg, -5.0)
        # 缓存里的 reg 三个目标**全部在 log10 空间**（实测 POR p50 0.99≈log10(9.8%)、
        # SW 1.93≈log10(85)、PERM -0.893），不是 src/feature.py 的 perm_log 语义。
        # 与 diag_seg_bottleneck.py:73 的 np.power(10.0, reg_blend) 一致。
        pred_real[c] = np.power(10.0, reg)
        y_true[c] = Y[c].to_numpy().astype(float)
        is_def_true[c] = DM[c].to_numpy()

    def score(th: dict) -> tuple[float, dict]:
        """给定逐目标标量阈值，算官方 total 与三目标 acc（全 8 井逐点等权）。"""
        acc = {}
        for c in TARGETS:
            p = np.where(pm[c] >= th[c],
                         np.full_like(y_true[c], DEFAULT_VALUES[c]),
                         pred_real[c])
            a = acc_vec(c, p, y_true[c])
            a = np.where(np.isfinite(y_true[c]), a, np.nan)
            acc[c] = np.nanmean(a)
        total = 100.0 * sum(WEIGHTS[c] * acc[c] for c in TARGETS)
        return total, acc

    # ── 1. 自检：必须复现 v7 的 73.749 ──
    tot0, acc0 = score(V7["thresholds"])
    print("=" * 78)
    print(f"[自检] v7 阈值 {V7['thresholds']}")
    print(f"       Total = {tot0:.3f}   POR {acc0['POR']:.4f}  PERM {acc0['PERM']:.4f}  SW {acc0['SW']:.4f}")
    print(f"       预期 73.749 ± 0.05 → {'通过' if abs(tot0 - 73.749) <= 0.05 else '!! 不通过，后续 Δ 不可信'}")
    print(f"       用时 {time.time() - t0:.1f}s")

    # ── 2. 真值井级默认占比：先看井间差异有多大 ──
    print("\n" + "=" * 78)
    print("[真值] 逐井默认段占比（这是井级常数，全局阈值无法同时对齐）")
    uniq = []
    for w in va["logId"].unique():
        if w not in uniq:
            uniq.append(w)
    uniq = sorted(uniq)
    for w in uniq:
        sl = well == w
        f = "  ".join(f"{c} {is_def_true[c][sl].mean():.3f}" for c in TARGETS)
        print(f"   {w[:12]}  n={sl.sum():>6}   {f}")

    # ── 3. ORACLE：已知真值占比，每井每目标独立选阈值 ──
    #    两种选法：(a) 使预测占比 == 真值占比（transductive 方法会做的）
    #              (b) 直接最大化该井该目标的官方 acc（该方向的真正上界）
    grid = np.linspace(0.01, 0.995, 970)

    def per_well_oracle(mode: str) -> tuple[float, dict]:
        acc = {}
        for c in TARGETS:
            v = np.full(len(va), np.nan)
            for w in uniq:
                sl = well == w
                idx = np.flatnonzero(sl)
                yt = y_true[c][idx]
                d = is_def_true[c][idx]
                pv = pm[c][idx]
                good = np.isfinite(yt)
                if good.sum() == 0:
                    continue
                if mode == "match":
                    # transductive 方法会做的事：让该井的预测默认占比 == 真值占比。
                    # 分位数网格上取最接近的阈值。
                    f = d[good].mean()
                    qs = np.unique(pv[good])
                    errs = np.abs((pv[good] >= qs[:, None]).mean(axis=1) - f)
                    t = float(qs[int(np.argmin(errs))])
                    p = np.where(pv >= t, DEFAULT_VALUES[c], pred_real[c][idx])
                    v[idx] = acc_vec(c, p, yt)
                else:
                    best, bt = -1.0, grid[0]
                    for t in grid:
                        p = np.where(pv >= t, DEFAULT_VALUES[c], pred_real[c][idx])
                        s = np.nanmean(acc_vec(c, p, yt))
                        if s > best:
                            best, bt = s, t
                    v[idx] = acc_vec(c, np.where(pv >= bt, DEFAULT_VALUES[c], pred_real[c][idx]), yt)
            acc[c] = float(np.nanmean(v))
        total = 100.0 * sum(WEIGHTS[c] * acc[c] for c in TARGETS)
        return total, acc

    tot_match, acc_match = per_well_oracle("match")
    tot_best, acc_best = per_well_oracle("best")
    print("=" * 78)
    print("[ORACLE] 逐井阈值（真值占比已知 —— 不可达上界，仅用于判断方向价值）")
    for mode, tot, acc in (("占比对齐", tot_match, acc_match), ("该井最优", tot_best, acc_best)):
        s = "  ".join(f"{c} {acc[c]:.4f}" for c in TARGETS)
        print(f"   {mode:<6} Total = {tot:.3f}   {s}   Δv7 = {tot - tot0:+.3f}")

    # ── 4. 全局阈值 vs 逐井阈值：差异有多大 ──
    print("=" * 78)
    print("[对比] v7 全局阈值下，各井的预测占比 vs 真值占比（系统性偏差）")
    for c in TARGETS:
        th = V7["thresholds"][c]
        pred_f = (pm[c] >= th).mean()
        true_f = is_def_true[c].mean()
        print(f"   {c:<5} 阈值 {th:.3f}  预测占比 {pred_f:.4f}  真值 {true_f:.4f}  "
              f"偏差 {pred_f - true_f:+.4f}")

    print(f"\n总用时 {time.time() - t0:.1f}s")
    print("判断标准：ORACLE Δ > +0.5 才值得建无监督占比估计器；< +0.1 则方向当场关闭。")


if __name__ == "__main__":
    main()
