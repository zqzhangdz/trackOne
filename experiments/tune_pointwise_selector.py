"""点级模型选择器：同一点上不同候选各能得多少分？可否被小型元分类器学到？

背景：换回归器加权方案**完全不动分**（4 种加权 POR acc 均 0.5933），
说明瓶颈不在回归器损失加权，而在**同一点上不同候选的误差相差很大**。

候选集 = 默认常数 + 7 个 (树, TCN) 混合比。逐点 oracle = 所有候选的最大分。
元分类器 = HistGB 在候选上 argmax。

**诚实口径**：选择器只在 5 口 screen 井上拟合，**只在留出 3 口井上报告**，
避免自欺（在训练集上评估会把增益显著高估）。v6 基线与 oracle 上界在全部 8 口上算。

内存策略（16 GB 机器）：只载入 8 口验证井、全程 float32、逐候选列计算后释放、
每目标结束 del + gc.collect()、HistGB max_iter=100。
只用 screen 缓存前向，不重训 TCN、不 import torch。
用法: python experiments/tune_pointwise_selector.py
"""
from __future__ import annotations
import gc, os, sys, warnings, time, json
warnings.filterwarnings("ignore")
import joblib, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sklearn.ensemble import HistGradientBoostingClassifier
from src.feature import load_wells, make_targets
from src.utils import DEFAULT_VALUES, EPS, TARGETS, WEIGHTS

RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
N_VAL_WELLS = 8
N_HOLDOUT_WELLS = 3
TCN_KEY = {"T15": "T15_seed42_150ep", "T16a": "T16_seed43_150ep", "T16b": "T16_seed44_150ep"}
TCN = list(TCN_KEY)
V6 = {"blend_w": {"POR": 0.54, "PERM": 0.96, "SW": 0.80},
      "thresholds": {"POR": 0.115, "PERM": 0.065, "SW": 0.040},
      "clf_tcn_w": {"POR": 0.60, "PERM": 0.75, "SW": 0.30}}
# 候选 0 = 默认常数，候选 1..7 = (树, TCN) 混合比（TCN 份额）
BLENDS = np.float32([0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0])
NCAND = len(BLENDS) + 1


def acc_official(c, pred, true):
    p = np.asarray(pred, np.float64); t = np.asarray(true, np.float64)
    if c == "PERM":
        return np.float32(np.maximum(0.0, 1.0 - np.abs(np.log10(np.maximum(p / np.maximum(t, EPS), EPS)))))
    tol = 0.08 if c == "POR" else 0.05
    return np.float32(np.maximum(0.0, 1.0 - np.abs(p - t) / (tol * (np.abs(t) + EPS))))


def main() -> None:
    t0 = time.time()
    wells = load_wells("./train", has_labels=True)
    ids = [w["logId"].iloc[0] for w in wells]
    val_ids = ids[80 - N_VAL_WELLS:]
    frames = [w for w in wells if w["logId"].iloc[0] in set(val_ids)]
    va = pd.concat(frames, ignore_index=True)
    del wells, frames
    gc.collect()
    Y = make_targets(va)
    # 留出井的掩码（按井边界切，最后 3 口留出）
    lens = [int((va["logId"] == i).sum()) for i in val_ids]
    keep = N_VAL_WELLS - N_HOLDOUT_WELLS
    bound = sum(lens[:keep])
    holdout = np.zeros(len(va), bool)
    holdout[bound:] = True
    Y = Y.reset_index(drop=True)
    del va
    gc.collect()

    cache = joblib.load(os.path.join(RUNS, "screen_tree_cache.joblib"))
    rt, pt = cache["reg"], cache["p_def"]
    del cache
    npz = np.load(os.path.join(RUNS, "tcn_screen_outputs.npz"))

    tot_v6_ho = tot_sel_ho = tot_or_ho = tot_v6_all = tot_oracle_all = 0.0
    print("=" * 98)
    print("点级选择器对照（screen 8 井；选择器仅在 5 井拟合，报告值取留出 3 井）")
    print("=" * 98)
    out = {}
    for c in TARGETS:
        gc.collect()
        tree = np.float32(rt[c])
        ens = np.float32(np.mean([npz[f"{TCN_KEY[t]}__{c}__reg"] for t in TCN], axis=0))
        pcn = np.float32(np.mean([npz[f"{TCN_KEY[t]}__{c}__pdf"] for t in TCN], axis=0))
        pm = np.float32((1 - V6["clf_tcn_w"][c]) * pt[c] + V6["clf_tcn_w"][c] * pcn)
        y = np.float64(Y[c].to_numpy())
        m = ~np.isnan(y)
        yl = y[m]
        d_log = np.log10(np.maximum(DEFAULT_VALUES[c], EPS))
        tree_m, ens_m, pm_m = tree[m], ens[m], pm[m]
        ho_m = holdout[m]            # 留出井掩码（与 m 同长）
        N = int(m.sum())

        # 逐候选列计算并只保留当前最优（避免 (N, C) 矩阵）
        best_acc = np.full(N, -1.0, np.float32)
        best_idx = np.zeros(N, np.int32)

        def consider(acc, j):
            up = acc > best_acc
            np.copyto(best_acc, np.where(up, acc, best_acc))
            np.copyto(best_idx, np.where(up, j, best_idx))

        consider(acc_official(c, np.full(N, DEFAULT_VALUES[c], np.float64), yl), 0)
        for j, w in enumerate(BLENDS, start=1):
            consider(acc_official(c, np.power(10.0, np.float32((1 - w) * tree_m + w * ens_m)), yl), j)
            gc.collect()
        a_oracle = float(best_acc.mean())

        # v6 基线：阈值判默认 + 融合权重
        is_def = pm_m >= V6["thresholds"][c]
        pred_v6 = np.where(is_def, DEFAULT_VALUES[c],
                           np.power(10.0, np.float32((1 - V6["blend_w"][c]) * tree_m + V6["blend_w"][c] * ens_m)))
        a_v6 = float(acc_official(c, pred_v6, yl).mean())

        # 元分类器：选 8 个候选之一
        feats = np.float32(np.column_stack([
            tree_m, ens_m, pm_m, np.abs(tree_m - ens_m), tree_m - ens_m,
            np.abs(tree_m), pm_m * pm_m,
        ]))
        tr = ~ho_m
        clf = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.1,
                                             max_leaf_nodes=31, random_state=42)
        clf.fit(feats[tr], best_idx[tr])
        pick = clf.predict(feats)   # feats 已是 N 长
        cand = np.zeros((NCAND, N), np.float32)
        cand[0] = d_log
        for j, w in enumerate(BLENDS, start=1):
            cand[j] = np.float32((1 - w) * tree_m + w * ens_m)
        pred_sel = np.power(10.0, cand[pick, np.arange(N)])
        hm = ho_m
        n_ho = int(hm.sum())
        a_sel = float(acc_official(c, pred_sel[~hm], yl[~hm]).mean())          # 拟合井（乐观）
        a_sel_ho = float(acc_official(c, np.power(10.0, cand[pick[hm], np.arange(n_ho)]), yl[hm]).mean())
        a_v6_ho = float(acc_official(c, pred_v6[hm], yl[hm]).mean())
        a_or_ho = float(best_acc[hm].mean())
        a_or_all = float(best_acc.mean())

        del feats, clf, cand, pred_v6, pred_sel, tree, ens, pcn, pm, y, yl, best_acc, best_idx
        gc.collect()

        tot_v6_ho += WEIGHTS[c] * a_v6_ho
        tot_sel_ho += WEIGHTS[c] * a_sel_ho
        tot_or_ho += WEIGHTS[c] * a_or_ho
        tot_v6_all += WEIGHTS[c] * a_v6
        tot_oracle_all += WEIGHTS[c] * a_or_all
        out[c] = {"v6_holdout": round(a_v6_ho, 4), "selector_holdout": round(a_sel_ho, 4),
                  "selector_fitted": round(a_sel, 4), "oracle_holdout": round(a_or_ho, 4)}
        print(f"  {c:5s} 留出3井: v6={a_v6_ho:.4f}  选择器={a_sel_ho:.4f} "
              f"(Δ{(a_sel_ho-a_v6_ho)*100:+.3f}分)  oracle={a_or_ho:.4f} "
              f"(Δ{(a_or_ho-a_v6_ho)*100:+.3f}分)")
        print(f"         选择器捕获 oracle 的 {(a_sel_ho-a_v6_ho)/max(a_or_ho-a_v6_ho,1e-9):.0%}"
              f"   选择器拟合井乐观值={a_sel:.4f}（比留出高 {100*(a_sel-a_sel_ho):+.3f} 分）"
              f"  [{time.time()-t0:.0f}s]")

    print("\n" + "=" * 98)
    print(f"  Total（留出 3 井同口径）  v6 = {tot_v6_ho*100:.3f}"
          f"  选择器 = {tot_sel_ho*100:.3f} (Δ{(tot_sel_ho-tot_v6_ho)*100:+.3f}分)"
          f"  oracle = {tot_or_ho*100:.3f} (Δ{(tot_or_ho-tot_v6_ho)*100:+.3f}分)")
    print(f"  选择器捕获 oracle 增益的 {(tot_sel_ho-tot_v6_ho)/max(tot_or_ho-tot_v6_ho,1e-9):.0%}"
          f"   缺口 {75.0-tot_sel_ho*100:+.3f}")
    print(f"  Total（全部 8 井）        v6 = {tot_v6_all*100:.3f}"
          f"  oracle = {tot_oracle_all*100:.3f} (Δ{(tot_oracle_all-tot_v6_all)*100:+.3f}分)")
    print("=" * 98)
    with open(os.path.join(RUNS, "pointwise_selector.json"), "w", encoding="utf-8") as f:
        json.dump({"v6_holdout": tot_v6_ho, "selector_holdout": tot_sel_ho, "oracle_holdout": tot_or_ho,
                   "v6_all": tot_v6_all, "oracle_all": tot_oracle_all, "per_target": out},
                  f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
