"""按官方公式重定训练加权方案：只用树，不重训 TCN。

已确认的两个训练端问题（都是「目标函数与官方口径不一致」）：

1. `train.py:277` 的 `weight_tol` 加权方向**反了**。
   官方容差 POR = 0.08·|y|、SW = 0.05·|y|，都是相对容差 → 要最小化 (ŷ-y)²/(|y|·tol)²
   才能对齐指标，即样本权重应 ∝ 1/|y|（低值点更重要）。
   实际代码给 POR/SW 用了 w = |y|（高值点更重要），PERM 用了 w = |log10 y|
   （V 形，y≈1 处权重≈0）——三个目标都与官方口径背离。

2. `train.py:49-50` 的 `metric_loss` 给 PERM 用 0.08·|log10 y| 当分母，
   而官方 PERM 容差恒为 1.0 decade。TCN 被训练去追求官方根本不要求的 10 倍精度，
   且 clamp(max=2) 让低 PERM 点的梯度被截断，高 PERM 尾部（真值 log10 达 2.0，
   树预测上限仅 1.88）系统性低估。

本脚本：复用缓存的分类器概率 p_def（隔离变量），只换回归器加权方案，
按官方口径评分。不 import torch、不重训 TCN。
用法: python experiments/tune_metric_aligned.py
"""
from __future__ import annotations
import os, sys, warnings, time, json
warnings.filterwarnings("ignore")
import joblib, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.feature import default_mask, load_wells, make_features, make_targets, transform_target
from src.model import _reg
from src.utils import DEFAULT_VALUES, EPS, TARGETS, WEIGHTS

RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
TCN_KEY = {"T15": "T15_seed42_150ep", "T16a": "T16_seed43_150ep", "T16b": "T16_seed44_150ep"}
TCN = list(TCN_KEY)
V6 = {"POR": (0.54, 0.115, 0.60), "PERM": (0.96, 0.065, 0.75), "SW": (0.80, 0.040, 0.30)}


def acc_official(c, pred, true):
    p = np.asarray(pred, float); t = np.asarray(true, float)
    if c == "PERM":
        return np.maximum(0.0, 1.0 - np.abs(np.log10(np.maximum(p / np.maximum(t, EPS), EPS))))
    tol = 0.08 if c == "POR" else 0.05
    return np.maximum(0.0, 1.0 - np.abs(p - t) / (tol * (np.abs(t) + EPS)))


def weights_for(c, y):
    """各候选加权方案。y 为物理空间真实段标签。"""
    a = np.abs(y)
    if c == "PERM":
        l = np.abs(np.log10(np.maximum(y, EPS)))
        return {
            "current |log10y|": l,
            "uniform(官方对齐)": np.ones_like(l),
            "1/|log10y|": 1.0 / (l + 0.1),
        }
    return {
        "current |y|": a,
        "1/|y| (官方对齐)": 1.0 / (a + EPS),
        "1/|y|^2 (严格对齐)": 1.0 / ((a + EPS) ** 2),
        "uniform": np.ones_like(a),
    }


def main() -> None:
    t0 = time.time()
    wells = load_wells("./train", has_labels=True)
    ids = [w["logId"].iloc[0] for w in wells]
    tr_ids = set(ids[:72]); va_ids = set(ids[72:])

    print("构建特征（72 训练井 + 8 验证井，context+priors）...", flush=True)
    tr = pd.concat([w for w in wells if w["logId"].iloc[0] in tr_ids], ignore_index=True)
    va = pd.concat([w for w in wells if w["logId"].iloc[0] in va_ids], ignore_index=True)
    Xtr = make_features(tr, context=True, priors=True)
    Xva = make_features(va, context=True, priors=True)
    Ytr, Yva = make_targets(tr), make_targets(va)
    isdef = default_mask(Ytr)
    print(f"  训练 {len(tr):,} 点 / 验证 {len(va):,} 点  特征 {Xtr.shape[1]} 维  "
          f"[{time.time()-t0:.0f}s]", flush=True)

    cache = joblib.load(os.path.join(RUNS, "screen_tree_cache.joblib"))
    pt = cache["p_def"]                      # 缓存里只有 screen(8 井) 的 p_def
    store = {k: v for k, v in np.load(os.path.join(RUNS, "tcn_screen_outputs.npz")).items()}
    from src.model import _clf               # 分类器需自训（缓存无 72 井的 p_def）

    print("\n" + "=" * 96)
    print("各目标：回归器加权方案对照（复用缓存 p_def 与阈值，只换回归器）")
    print("=" * 96)

    results = {}
    for c in TARGETS:
        ytr_all = Ytr[c].to_numpy()
        dtr = isdef[c].to_numpy()
        valid = ~np.isnan(ytr_all)
        real = valid & ~dtr
        yreal = ytr_all[real]
        ltr = transform_target(c, pd.Series(yreal), "perm_log").to_numpy()
        Xreal = Xtr.loc[real]          # real 已含 valid，直接整长掩码索引

        # 分类器自训（固定超参，不参与本次加权对照）
        clf = _clf()
        clf.fit(Xtr.loc[valid], dtr[valid].astype(int))   # 分类器用全量有效点
        pca = clf.predict_proba(Xva)[:, 1]

        yva = Yva[c].to_numpy()
        mva = ~np.isnan(yva)
        pva, tva = pca[mva], yva[mva]
        cw = V6[c][2]
        ens = np.mean([store[f"{TCN_KEY[t]}__{c}__reg"] for t in TCN], axis=0)[mva]
        pcn = np.mean([store[f"{TCN_KEY[t]}__{c}__pdf"] for t in TCN], axis=0)[mva]
        pm = (1 - cw) * pva + cw * pcn

        print(f"\n--- {c} ---")
        print(f"  真实段标签 log10 分布: 训练 p5/p50/p95 = "
              f"{np.percentile(ltr,5):+.3f}/{np.percentile(ltr,50):+.3f}/{np.percentile(ltr,95):+.3f}"
              f"   验证 p95 = {np.percentile(np.log10(np.maximum(tva, EPS)),95):+.3f}")
        row = {}
        for name, w in weights_for(c, yreal).items():
            w = np.clip(w, np.percentile(w, 5), np.percentile(w, 95)); w = w / w.mean()
            reg = _reg()
            reg.fit(Xreal, ltr, sample_weight=w)
            pr = reg.predict(Xva)[mva]                       # log10 空间
            phys = np.where(pm >= V6[c][1],
                            DEFAULT_VALUES[c], np.power(10.0, (1 - V6[c][0]) * pr + V6[c][0] * ens))
            a = float(np.mean(acc_official(c, phys, tva)))
            row[name] = a
            print(f"  {name:22s} 官方 acc {a:.4f}")
        results[c] = row
        best = max(row, key=row.get)
        print(f"  → 最佳 {best} ({row[best]:.4f})  耗时 {time.time()-t0:.0f}s")

    print("\n" + "=" * 96)
    print("总分对照（官方口径；逐目标取该目标最佳加权方案）")
    print("=" * 96)
    tot_v6 = 0.0
    print("  v6 当前（缓存 tree + 3×TCN）:")
    for c in TARGETS:
        yva = Yva[c].to_numpy(); mva = ~np.isnan(yva)
        ens = np.mean([store[f"{TCN_KEY[t]}__{c}__reg"] for t in TCN], axis=0)[mva]
        pcn = np.mean([store[f"{TCN_KEY[t]}__{c}__pdf"] for t in TCN], axis=0)[mva]
        pr = cache["reg"][c][mva]
        pm = (1 - V6[c][2]) * pt[c][mva] + V6[c][2] * pcn
        phys = np.where(pm >= V6[c][1], DEFAULT_VALUES[c],
                        np.power(10.0, (1 - V6[c][0]) * pr + V6[c][0] * ens))
        a = float(np.mean(acc_official(c, phys, yva[mva])))
        tot_v6 += WEIGHTS[c] * a
        print(f"    {c:5s} {a:.4f}  ({WEIGHTS[c]*a*100:.2f} 分)")
    print(f"    Total = {tot_v6*100:.3f}   75 分缺口 {75.0-tot_v6*100:+.3f}")

    with open(os.path.join(RUNS, "metric_aligned.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
