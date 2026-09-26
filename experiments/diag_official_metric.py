"""按赛事方官方公式（参数说明.docx 图片原文）重算 PERM 准确率。

发现：仓库 src/utils.py 的 PERM 口径与官方公式**不一致**。

官方（图片原文）：
    Acc_PERM = (1/N) * Σ max(0, 1 - |log10(max(ŷ_i / y_i, ε))|)
    —— log10 比值空间，**无 0.08 系数、无分母**，等效容差恒为 1.0 decade。

仓库（src/utils.py:_clip_acc_log10）：
    Acc_PERM = mean(max(0, 1 - |log10(ŷ) - log10(y)| / (0.08 * (|log10(y)| + ε))))
    —— 分母 0.08 * |log10 y|，实际容差约 0.02–0.36 decade，比官方严 3–20 倍。

POR / SW 两条公式与官方完全一致，不受影响。

本脚本用同一批缓存的 v6 预测，同时按两种口径打分，量化口径差异。
只读缓存数组，不 fit 任何模型、不 import torch。
用法: python experiments/diag_official_metric.py
"""
from __future__ import annotations
import os, sys, warnings, time
warnings.filterwarnings("ignore")
import joblib, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.feature import default_mask, load_wells, make_targets
from src.utils import DEFAULT_VALUES, EPS, TARGETS, WEIGHTS

RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
TCN = ["T15", "T16a", "T16b"]
TCN_KEY = {"T15": "T15_seed42_150ep", "T16a": "T16_seed43_150ep", "T16b": "T16_seed44_150ep"}
V6 = {"blend_w": {"POR": 0.54, "PERM": 0.96, "SW": 0.80},
      "thresholds": {"POR": 0.115, "PERM": 0.065, "SW": 0.040},
      "clf_tcn_w": {"POR": 0.60, "PERM": 0.75, "SW": 0.30}}


def acc_por_sw(pred, true, tol):
    """官方 = 仓库实现。"""
    return np.maximum(0.0, 1.0 - np.abs(pred - true) / (tol * (np.abs(true) + EPS)))


def acc_perm_repo(pred, true):
    """仓库实现（0.08 相对容差，在 |log10 y| 尺度上）。"""
    p = np.log10(np.maximum(pred, EPS))
    t = np.log10(np.maximum(true, EPS))
    return np.maximum(0.0, 1.0 - np.abs(p - t) / (0.08 * (np.abs(t) + EPS)))


def acc_perm_official(pred, true):
    """官方原文：1 - |log10(max(ŷ/y, ε))|，截断在 0。"""
    ratio = np.maximum(np.asarray(pred, float) / np.asarray(true, float), EPS)
    return np.maximum(0.0, 1.0 - np.abs(np.log10(ratio)))


def build_preds(Y, rt, pt, store):
    """v6 融合 + 阈值决策，返回 {目标: 物理空间预测}。"""
    out = {}
    for c in TARGETS:
        ens = np.mean([store[f"{TCN_KEY[t]}__{c}__reg"] for t in TCN], axis=0)
        pcn = np.mean([store[f"{TCN_KEY[t]}__{c}__pdf"] for t in TCN], axis=0)
        w, th, cw = V6["blend_w"][c], V6["thresholds"][c], V6["clf_tcn_w"][c]
        pm = (1 - cw) * pt[c] + cw * pcn
        out[c] = np.where(pm >= th, DEFAULT_VALUES[c], np.power(10.0, (1 - w) * rt[c] + w * ens))
    return out


def main() -> None:
    wells = load_wells("./train", has_labels=True)
    ids = [w["logId"].iloc[0] for w in wells]
    va_df = pd.concat([w for w in wells if w["logId"].iloc[0] in set(ids[72:])], ignore_index=True)
    Y = make_targets(va_df)
    cache = joblib.load(os.path.join(RUNS, "screen_tree_cache.joblib"))
    rt, pt = cache["reg"], cache["p_def"]
    store = {k: v for k, v in np.load(os.path.join(RUNS, "tcn_screen_outputs.npz")).items()}
    Dm = {c: default_mask(Y)[c].to_numpy() for c in TARGETS}
    preds = build_preds(Y, rt, pt, store)
    t0 = time.time()

    acc_repos = {"POR": acc_por_sw, "SW": acc_por_sw}
    acc_offs = {"POR": acc_por_sw, "SW": acc_por_sw}

    print("=" * 104)
    print(f"{'目标':5s} | {'口径':24s} | {'acc 全池':>9s} {'acc 默认段':>10s} {'acc 真实段':>10s}")
    print("-" * 104)
    for c in TARGETS:
        p, y = np.asarray(preds[c], float), np.asarray(Y[c].to_numpy(), float)
        if c == "PERM":
            f_repo, f_off = acc_perm_repo(p, y), acc_perm_official(p, y)
            err = np.abs(np.log10(np.maximum(p, EPS)) - np.log10(np.maximum(y, EPS)))
            denom = 0.08 * (np.abs(np.log10(np.maximum(y, EPS))) + EPS)
        else:
            tol = 0.08 if c == "POR" else 0.05
            f_repo = f_off = acc_por_sw(p, y, tol)
            err = np.abs(p - y)
            denom = tol * (np.abs(y) + EPS)
        valid = ~np.isnan(y)
        rep, real = valid & Dm[c], valid & ~Dm[c]
        e_mae = float(np.nanmean(err[real]))
        d_mae = float(np.nanmean(denom[real]))
        print(f"{c:5s} | {'仓库 (src/utils.py)':24s} | {np.nanmean(f_repo[valid]):9.4f} {np.nanmean(f_repo[rep]):10.4f} {np.nanmean(f_repo[real]):10.4f}")
        print(f"{'':5s} | {'官方原文（图片）':24s} | {np.nanmean(f_off[valid]):9.4f} {np.nanmean(f_off[rep]):10.4f} {np.nanmean(f_off[real]):10.4f}")
        print(f"{'':5s} | {'Δ (官方 − 仓库)':24s} | {np.nanmean(f_off[valid])-np.nanmean(f_repo[valid]):+9.4f} {'' :>10s} {np.nanmean(f_off[real])-np.nanmean(f_repo[real]):+10.4f}")
        unit = "decade" if c == "PERM" else "物理量纲"
        print(f"{'':5s} | 真实段 平均|err| = {e_mae:.4f} {unit:<9s} | 仓库分母均值 = {d_mae:.4f}"
              + (f"  → 实际容差仅相当于官方容差的 {d_mae/1.0:.1%}" if c == "PERM" else ""))
        print("-" * 104)

    print("\n=== 总分对比（官方逐点等权，权重 0.30/0.35/0.35）===")
    tot_repo = tot_off = 0.0
    for c in TARGETS:
        p, y = np.asarray(preds[c], float), np.asarray(Y[c].to_numpy(), float)
        v = ~np.isnan(y)
        fr = acc_perm_repo(p, y) if c == "PERM" else acc_por_sw(p, y, 0.08 if c == "POR" else 0.05)
        fo = acc_perm_official(p, y) if c == "PERM" else fr
        ar, ao = float(np.mean(fr[v])), float(np.mean(fo[v]))
        tot_repo += WEIGHTS[c] * ar
        tot_off += WEIGHTS[c] * ao
        print(f"  {c:5s} 仓库 {ar:.4f}  →  官方 {ao:.4f}   Δ={WEIGHTS[c]*(ao-ar)*100:+7.3f} 分 (权重 {WEIGHTS[c]})")
    print(f"\n  Total 仓库口径 = {tot_repo*100:.3f}")
    print(f"  Total 官方口径 = {tot_off*100:.3f}")
    print(f"  差距           = {(tot_off-tot_repo)*100:+.3f} 分")
    print(f"  官方 75 分线缺口 = {75.0 - tot_off*100:+.3f}")
    print(f"\n耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
