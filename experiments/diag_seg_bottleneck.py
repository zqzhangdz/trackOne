"""瓶颈定位：分段分类器，不是回归器（本项目的核心结论，必须可复现）。

结论（screen 8 井，官方口径，逐点等权）：
  v6 现部署              = 73.310
  真值分段 + 同回归器    = 76.308   ← 分段器可追 +2.997 分
  75 分线缺口            = +1.690   → 空间是缺口的 1.77 倍

"同回归器"的含义：blend_w / clf_tcn_w / 默认常数全部照抄 v6，
只把"pm >= threshold 判默认段"这一步换成真值 default_mask。
所以这个 +2.997 是**纯粹的分段误差**，不含任何回归器改进。

同时输出分类器 AUC：AUC 只有 0.89(POR/PERM) / 0.97(SW)，且树与 TCN 几乎相同，
说明 76.308 不是可达上界，且现有两路融合提不了 AUC。

只用缓存前向；不 fit 模型、不 import torch（遵守同进程运行时约束）。
用法: python experiments/diag_seg_bottleneck.py
"""
from __future__ import annotations
import os, sys, gc, json, time, warnings
warnings.filterwarnings("ignore")
import joblib, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sklearn.metrics import roc_auc_score
from src.feature import load_wells, make_targets, default_mask
from src.utils import DEFAULT_VALUES, EPS, TARGETS, WEIGHTS

RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
T108 = {"T15": "T15_seed42_150ep", "T16a": "T16_seed43_150ep", "T16b": "T16_seed44_150ep"}
V6 = {"blend_w": {"POR": 0.54, "PERM": 0.96, "SW": 0.80},
      "thresholds": {"POR": 0.115, "PERM": 0.065, "SW": 0.040},
      "clf_tcn_w": {"POR": 0.60, "PERM": 0.75, "SW": 0.30}}


def acc_vec(c, pred, true):
    """官方口径逐点准确率向量（图片原文）。PERM 恒 1.0 decade 比值容差。"""
    p = np.asarray(pred, float); t = np.asarray(true, float)
    if c == "PERM":
        return np.maximum(0.0, 1.0 - np.abs(np.log10(np.maximum(p / np.maximum(t, EPS), EPS))))
    tol = 0.08 if c == "POR" else 0.05
    return np.maximum(0.0, 1.0 - np.abs(p - t) / (tol * (np.abs(t) + EPS)))


def main() -> None:
    t0 = time.time()
    wells = load_wells("./train", has_labels=True)
    ids = [w["logId"].iloc[0] for w in wells]
    va = pd.concat([w for w in wells if w["logId"].iloc[0] in set(ids[72:])], ignore_index=True)
    del wells; gc.collect()
    Y = make_targets(va)
    DM = default_mask(Y)

    cache = joblib.load(os.path.join(RUNS, "screen_tree_cache.joblib"))
    rt, pt = cache["reg"], cache["p_def"]
    store = {k: v for k, v in np.load(os.path.join(RUNS, "tcn_screen_outputs.npz")).items()}

    # 注意：不能链式 = {}，那样 5 个名字会指向同一个 dict（键互相覆盖）
    acc_dep, acc_or, f_def, acc_def, frac_pred = {}, {}, {}, {}, {}
    auc = {}
    for c in TARGETS:
        w, th, cw = V6["blend_w"][c], V6["thresholds"][c], V6["clf_tcn_w"][c]
        ens = np.mean([store[f"{T108[t]}__{c}__reg"] for t in T108], axis=0)   # log10 空间
        pcn = np.mean([store[f"{T108[t]}__{c}__pdf"] for t in T108], axis=0)
        p_tree, p_tcn = pt[c], pcn
        pm = (1 - cw) * p_tree + cw * p_tcn
        reg_blend = (1 - w) * rt[c] + w * ens

        y = Y[c].to_numpy()
        m = ~np.isnan(y)
        d = DM[c].to_numpy()
        n = int(m.sum())

        # 现部署：学习到的分段
        pred_dep = np.where(pm >= th, DEFAULT_VALUES[c], np.power(10.0, reg_blend))
        # 真值分段 + 同一回归器 + 同一默认常数
        pred_or = np.where(d, DEFAULT_VALUES[c], np.power(10.0, reg_blend))

        av = acc_vec(c, pred_dep, y); aov = acc_vec(c, pred_or, y)
        a_def_v = acc_vec(c, np.full(len(y), DEFAULT_VALUES[c]), y)
        acc_dep[c] = float(av[m].mean())
        acc_or[c] = float(aov[m].mean())
        f_def[c] = float((m & d).sum() / n)
        acc_def[c] = float(a_def_v[m & d].mean())
        frac_pred[c] = float((pm >= th)[m].mean())

        # 分类器 AUC：正类 = 默认段（与 p_def 同向），AUC 越高越好
        yb = (m & d).astype(int)[m]
        for name, ps in (("tree", p_tree), ("tcn", p_tcn), ("v6blend", pm)):
            auc[(c, name)] = float(roc_auc_score(yb, ps[m]))

        del ens, pcn, reg_blend, pred_dep, pred_or, av, aov, a_def_v
        gc.collect()

    tot = lambda a: float(sum(WEIGHTS[c] * a[c] for c in TARGETS) * 100)
    print("=" * 104)
    print("瓶颈定位：分段分类器 vs 回归器（screen 8 井，官方口径，逐点等权）")
    print("=" * 104)
    print(f"{'目标':6s} {'v6 现部署':>10s} {'贡献':>8s} {'真值分段+同回归器':>18s} "
          f"{'贡献':>8s} {'分段器可追':>11s}")
    for c in TARGETS:
        print(f"{c:6s} {acc_dep[c]:10.4f} {WEIGHTS[c]*acc_dep[c]*100:8.3f} "
              f"{acc_or[c]:18.4f} {WEIGHTS[c]*acc_or[c]*100:8.3f} "
              f"{(WEIGHTS[c]*(acc_or[c]-acc_dep[c])*100):+11.3f}")
    td, to = tot(acc_dep), tot(acc_or)
    print("-" * 104)
    print(f"{'Total':6s} {td:10.3f} {'':>8s} {to:18.3f} {'':>8s} {to-td:+11.3f}")
    print(f"\n  v6 现部署总分        = {td:.3f}")
    print(f"  真值分段上界（同模型） = {to:.3f}   ← 分段器可追 {to-td:+.3f} 分")
    print(f"  75 分缺口            = {75.0-td:+.3f}")
    print(f"  可追/缺口            = {(to-td)/(75.0-td):.2f} 倍"
          f"（{'空间够' if (to-td) >= 75.0-td else '空间不够'}）")

    print("\n分解核对：acc = f_def·acc_def + (1-f_def)·acc_real")
    print(f"{'目标':6s} {'acc':>8s} {'f_def':>8s} {'acc_def':>9s} {'acc_real':>10s}")
    for c in TARGETS:
        ar = (acc_dep[c] - f_def[c] * acc_def[c]) / (1 - f_def[c])
        print(f"{c:6s} {acc_dep[c]:8.4f} {f_def[c]:8.4f} {acc_def[c]:9.4f} {ar:10.4f}")
    print("\n  注：此处 acc_def 恒为 1.0000 —— default_mask 的定义是 isclose(y, 默认值)，")
    print("      所以默认段点用默认常数预测必然满分。台账里的 0.9881/0.9941/0.9929 是用")
    print("      「部署模型判默认」的掩码算的，acc_real 相应为 0.1694/0.4015/0.1450。")
    print("      两种拆法 acc 与 Total 完全一致（都是 73.310），只是 real/def 边界取法不同。")

    print("\n" + "=" * 104)
    print("分类器 AUC（正类 = 默认段，与 p_def 同向）。AUC 低 = 信息不足，不是阈值没调好")
    print("=" * 104)
    print(f"{'目标':6s} {'AUC(树)':>9s} {'AUC(TCN)':>9s} {'AUC(v6融合)':>11s} "
          f"{'v6@th判默认占比':>16s} {'真值':>8s}")
    for c in TARGETS:
        print(f"{c:6s} {auc[(c,'tree')]:9.4f} {auc[(c,'tcn')]:9.4f} {auc[(c,'v6blend')]:11.4f} "
              f"{frac_pred[c]:16.4f} {f_def[c]:8.4f}")

    with open(os.path.join(RUNS, "seg_bottleneck.json"), "w", encoding="utf-8") as f:
        json.dump({"acc_deployed": acc_dep, "acc_oracle_seg": acc_or, "f_def": f_def,
                   "acc_def": acc_def, "frac_predicted_default": frac_pred,
                   "auc": {f"{c}|{k}": v for (c, k), v in auc.items()},
                   "total_deployed": td, "total_oracle_seg": to,
                   "recoverable": to - td, "gap_to_75": 75.0 - td},
                  f, ensure_ascii=False, indent=2)
    print(f"\n耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
