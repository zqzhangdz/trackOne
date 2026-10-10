"""联合（多目标池化）分段分类器：把多个目标的标签信号叠起来训练同一个分类器。

动机
----
§4.1 的默认段占比：POR 60.03% / PERM 60.63% / SW 73.40%。POR 与 PERM 几乎相同，
说明它们的分段边界高度一致。分段器是本项目的硬瓶颈（§4.2：真值分段上界 76.336，
现部署 73.931，全部 +2.4 都在这一步），而 §4.11 已证明**换训练方式**动不了 AUC
（0.897 是特征的信息上限）。

已否决的是「跨目标 p_def 特征」（把别的目标的 p_def 当本目标的特征喂进去）——那只在
**特征空间**里跨目标。本脚本做的是**标签池化**：把 POR 与 PERM 的行拼起来训练同一个
分类器（加一列 target-identity 让它能区分目标），每个目标看到的有监督样本从 20.8 万
变成 41.0 万。这与前者是不同的实验，从未试过。

判据
----
  1) 先测默认段掩码的两两一致率（若不一致率 > 1%，池化会互相污染，方向当场关闭）；
  2) 轻模型（100 轮）在验证 8 井上做 AUC 初筛，只有 AUC 明显上升的池化组合才做全量评分；
  3) 全量评分用**部署同源**口径（weight_tol 树 + 3 seed TCN，v9 融合），逐目标换 clf，
     看 Total 与分解表（acc / acc_def / acc_real）。

用法: python experiments/diag_joint_clf.py
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
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from src.feature import load_wells, make_targets, default_mask, make_features, inverse_transform_target
from src.utils import TARGETS, WEIGHTS, EPS, DEFAULT_VALUES
from experiments.eval_v4 import build_tree, tcn_outputs

RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
TAGS = ["T15_seed42_150ep", "T16_seed43_150ep", "T16_seed44_150ep"]
CFG = {"blend_w": {"POR": 0.50, "PERM": 0.50, "SW": 0.75},
       "thresholds": {"POR": 0.080, "PERM": 0.250, "SW": 0.050},
       "clf_tcn_w": {"POR": 0.60, "PERM": 0.70, "SW": 0.30}}
CLS = dict(learning_rate=0.1, max_iter=300, max_leaf_nodes=31,
           min_samples_leaf=50, l2_regularization=1.0, class_weight="balanced")


def acc_vec(c, p, t):
    p = np.asarray(p, float)
    t = np.asarray(t, float)
    if c == "PERM":
        return np.maximum(0.0, 1.0 - np.abs(np.log10(np.maximum(p / np.maximum(t, EPS), EPS))))
    tol = 0.08 if c == "POR" else 0.05
    return np.maximum(0.0, 1.0 - np.abs(p - t) / (tol * (np.abs(t) + EPS)))


def score_blend(reg_tree, ens, pt, pcn, Y, D, cls_c, ths=None):
    ths = ths or CFG["thresholds"]
    tot, parts = 0.0, {}
    for c in TARGETS:
        w, cw, th = CFG["blend_w"][c], CFG["clf_tcn_w"][c], ths[c]
        reg = (1 - w) * reg_tree[c] + w * ens[c]
        reg = np.where(np.isfinite(reg), reg, -5.0)
        pm = (1 - cw) * pt[c] + cw * pcn[c]
        if cls_c is not None and c in cls_c:
            pm = (1 - cw) * cls_c[c] + cw * pcn[c]
        y = Y[c].to_numpy().astype(float)
        valid = np.isfinite(y)
        pred = np.where(pm >= th, DEFAULT_VALUES[c], np.power(10.0, reg))
        a = np.where(np.isfinite(y), acc_vec(c, pred, y), np.nan)
        tot += WEIGHTS[c] * np.nanmean(a)
        d = D[c].to_numpy() & valid
        parts[c] = (float(np.nanmean(a)),
                    float(np.nanmean(a[valid & ~d])) if (valid & ~d).sum() else float("nan"),
                    float(np.nanmean(a[valid & d])) if (valid & d).sum() else float("nan"),
                    float(pm[valid].mean()))
    return tot * 100, parts


def main():
    t0 = time.time()
    wells = load_wells("./train", has_labels=True)
    ids = [w["logId"].iloc[0] for w in wells]
    by = dict(zip(ids, wells))
    tr = pd.concat([by[i] for i in ids[:72]], ignore_index=True)
    va = pd.concat([by[i] for i in ids[72:80]], ignore_index=True)
    del wells
    gc.collect()
    Ytr, Yva = make_targets(tr), make_targets(va)
    Dtr, Dva = default_mask(Ytr), default_mask(Yva)
    Xtr = make_features(tr, context=True, priors=True)
    Xva = make_features(va, context=True, priors=True)
    vtr = Ytr.notna().all(axis=1).to_numpy()
    dva = Yva.notna().all(axis=1).to_numpy()
    lab = {c: Dtr[c].to_numpy()[vtr].astype(int) for c in TARGETS}
    lvy = {c: Dva[c].to_numpy()[dva].astype(int) for c in TARGETS}
    print(f"训练 {len(tr):,} 点 / 验证 {len(va):,} 点   特征 {Xtr.shape[1]}")

    # ---------- 1) 默认段掩码一致率 ----------
    print("\n" + "=" * 92)
    print("默认段掩码两两一致率（训练 72 井）与默认段占比")
    print("=" * 92)
    print("        " + "".join(f"{c:>14}" for c in TARGETS))
    for a in TARGETS:
        row = f"{a:<8} "
        for b in TARGETS:
            row += f"{(lab[a] == lab[b]).mean() * 100:12.2f}% "
        print(row)
    print("        " + "".join(f"{lab[c].mean() * 100:12.2f}% " for c in TARGETS))
    agree_pp = float((lab["POR"] == lab["PERM"]).mean())
    agree_psw = float((lab["POR"] == lab["SW"]).mean())
    agree_msw = float((lab["PERM"] == lab["SW"]).mean())
    print(f"\n  POR↔PERM {agree_pp * 100:.2f}%   POR↔SW {agree_psw * 100:.2f}%   "
          f"PERM↔SW {agree_msw * 100:.2f}%")
    print("  不一致率 > 1% 的池化组合预期会互相污染；SW 与另两者的不一致率决定是否排除 SW")

    # ---------- 部署同源基线 ----------
    cache = joblib.load(os.path.join(RUNS, "screen_tree_cache.joblib"))
    tcn = {k: v for k, v in np.load(os.path.join(RUNS, "tcn_screen_outputs.npz")).items()}
    arts = [joblib.load(os.path.join(RUNS, f"tcn_{t}.joblib")) for t in TAGS]
    print(f"\n加载 3 个 TCN 工件并前向 8 验证井 …")
    ens, pcn = {}, {}
    for art in arts:
        r, p = tcn_outputs([by[i] for i in ids[72:80]], art)
        for c in TARGETS:
            ens.setdefault(c, []).append(r[c])
            pcn.setdefault(c, []).append(p[c])
    ens = {c: np.mean(v, axis=0) for c, v in ens.items()}
    pcn = {c: np.mean(v, axis=0) for c, v in pcn.items()}

    # 树的回归输出复用一次拟合（三个候选共用），分类器单独比较
    model, _ = build_tree(tr, weight_tol=True)
    reg_tree = {c: np.log10(np.maximum(inverse_transform_target(
        c, pd.Series(model.regressors[c].predict(Xva), index=Xva.index)).to_numpy(), EPS))
        for c in TARGETS}
    pt_cache = {c: cache["p_def"][c] for c in TARGETS}
    rt_cache = {c: cache["reg"][c] for c in TARGETS}
    # 自检：重拟合的树必须与缓存逐点一致（§4.10）
    for c in TARGETS:
        a = float(np.median(np.abs(np.sort(reg_tree[c]) - np.sort(rt_cache[c]))))
        b = float(np.median(np.abs(np.sort(pt_cache[c]) - np.sort(cache["p_def"][c]))))
        print(f"  [自检] {c} 树 reg {a:.2e} / p_def {b:.2e}"
              f"{'   ✓' if max(a, b) < 1e-6 else '   ✗'}")
        assert max(a, b) < 1e-6, "树/分类器与部署缓存不一致（§4.10 陷阱）"

    t_b, d_b = score_blend(rt_cache, ens, pt_cache, pcn, Yva, Dva, None)
    print(f"\n基线（缓存 p_def + 3 seed TCN，v9 口径）Total = {t_b:.3f}")
    for c in TARGETS:
        auc = float(roc_auc_score(lvy[c], pt_cache[c][dva]))
        print(f"  {c}  acc {d_b[c][0]:.4f}  real {d_b[c][1]:.4f}  def {d_b[c][2]:.4f}"
              f"  预测默认占比 {d_b[c][3] * 100:.1f}%（真值 {lvy[c].mean() * 100:.1f}%）"
              f"  AUC {auc:.4f}")

    # ---------- 2) AUC 初筛（轻模型，200 轮，全量样本）----------
    POOLS = {
        "POR": ("POR",),
        "PERM": ("PERM",),
        "SW": ("SW",),
        "POR+PERM": ("POR", "PERM"),
        "POR+SW": ("POR", "SW"),
        "PERM+SW": ("PERM", "SW"),
        "POR+PERM+SW": ("POR", "PERM", "SW"),
    }
    print("\n" + "=" * 92)
    print("AUC 初筛（HistGB 200 轮，加 1 列 target-identity）")
    print("=" * 92)
    light = dict(CLS, max_iter=200)
    screen = {}
    for name, pool in POOLS.items():
        if len(pool) == 1:
            continue
        Xs = np.concatenate([Xtr[vtr].to_numpy(np.float32) for _ in pool], axis=0)
        ys = np.concatenate([lab[c] for c in pool], axis=0)
        ti = np.concatenate([np.full(len(lab[c]), i, dtype=np.float32)
                             for i, c in enumerate(pool)], axis=0)
        Xs = np.column_stack([Xs, ti])
        cf = HistGradientBoostingClassifier(loss="log_loss", random_state=42, **light)
        cf.fit(Xs, ys)
        rec = {}
        for c in pool:
            rec[c] = float(roc_auc_score(lvy[c],
                       cf.predict_proba(np.column_stack([Xva[dva].to_numpy(np.float32),
                                                         np.full(dva.sum(), pool.index(c),
                                                                 dtype=np.float32)]))[:, 1]))
        screen[name] = rec
        base_auc = {c: float(roc_auc_score(lvy[c], pt_cache[c][dva])) for c in pool}
        print(f"  {name:<14} 池化 {len(ys):,} 行  " +
              "  ".join(f"{c} {base_auc[c]:.4f}→{rec[c]:.4f} ({rec[c] - base_auc[c]:+.4f})"
                        for c in pool))

    # ---------- 3) 全量评分（只对 AUC 上升的组合，用部署口径 300 轮重拟合）----------
    print("\n" + "=" * 92)
    print("全量评分（部署口径 clf 300 轮 + 部署同源回归器与 TCN，v9 融合）")
    print("=" * 92)
    for name, pool in POOLS.items():
        if len(pool) == 1 or name not in screen:
            continue
        base_auc = {c: float(roc_auc_score(lvy[c], pt_cache[c][dva])) for c in pool}
        gain = screen[name]
        if all(gain[c] <= base_auc[c] + 0.001 for c in pool):
            print(f"  {name:<14} 初筛无 AUC 增益 → 跳过（换训练方式提不了 AUC，§4.11）")
            continue
        Xs = np.concatenate([Xtr[vtr].to_numpy(np.float32) for _ in pool], axis=0)
        ys = np.concatenate([lab[c] for c in pool], axis=0)
        ti = np.concatenate([np.full(len(lab[c]), i, dtype=np.float32)
                             for i, c in enumerate(pool)], axis=0)
        Xs = np.column_stack([Xs, ti])
        cf = HistGradientBoostingClassifier(loss="log_loss", random_state=42, **CLS)
        cf.fit(Xs, ys)
        jc = {}
        for c in pool:
            jc[c] = cf.predict_proba(np.column_stack(
                [Xva[dva].to_numpy(np.float32),
                 np.full(dva.sum(), pool.index(c), dtype=np.float32)]))[:, 1]
        jc_full = {c: np.full(len(Yva), np.nan) for c in TARGETS}
        for c in pool:
            jc_full[c][dva] = jc[c]
        t1, d1 = score_blend(rt_cache, ens, jc_full, pcn, Yva, Dva, jc_full)
        print(f"  {name:<14} Total {t_b:.3f} → {t1:.3f}   Δ {t1 - t_b:+.3f} 分")
        for c in TARGETS:
            if c not in pool:
                continue
            print(f"     {c}  acc {d_b[c][0]:.4f}→{d1[c][0]:.4f}  "
                  f"real {d_b[c][1]:.4f}→{d1[c][1]:.4f}  def {d_b[c][2]:.4f}→{d1[c][2]:.4f}"
                  f"  默认占比 {d_b[c][3] * 100:.1f}→{d1[c][3] * 100:.1f}%"
                  f"  AUC {base_auc[c]:.4f}→"
                  f"{float(roc_auc_score(lvy[c], jc[c])):.4f}"
                  f"   Δ {(d1[c][0] - d_b[c][0]) * 100 * WEIGHTS[c]:+.4f} 分")

    print(f"\n用时 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
