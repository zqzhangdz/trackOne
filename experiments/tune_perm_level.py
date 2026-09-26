"""PERM 是唯一可达杠杆 —— 段级 PERM 预测能到多少？

结论来源（experiments/diag_official_ceiling.py，screen 8 井，官方口径）：
  POR  real 0.1694 vs 真值段均值 oracle 0.1666 → 已到顶，无空间
  SW   real 0.1450 vs oracle 0.2363            → 到顶，单靠它够不着 75
  PERM real 0.4015 vs oracle 0.8429            → 有空间！
  要够 75 分，PERM real 需 0.4015 → 0.5241（Δ+0.1226 = +1.69 分）

为什么之前判"PERM 不可达"：旧 PERM 容差 0.08·|log10 y| ≈ 0.02–0.36 decade，
而官方恒为 1.0 decade。段内 log10-RMSE 0.4154 对旧容差是"物理不可达"，
对官方容差完全在带内。

本轮测三个此前**未测过**的杠杆（旧口径下都被否决或没必要，官方口径下都值得重测）：
  1. loss='absolute_error' —— 官方 PERM 指标就是 log10 空间的 MAE，
     而 HistGB 默认 squared_error 预测的是**均值**，MAE 的最优解是**中位数**。
     这是之前用默认 loss 没试过的东西。
  2. 加入 POR —— PERM 在物理上是 POR + 岩性 的函数。加入真值 POR 是上界，
     加入树预测 POR 是可部署的。之前从没做过这个耦合。
  3. 点级（非段级）HistGB —— 段内 wiggle 能不能再加回来。

分段一律用【真值分段】（仅构造标签，评估逐点）。故本结果是"已知段边界"的上界。
内存：段级样本 ~数千行 × 50 维，无 torch，峰值 < 400 MB。
用法: python experiments/tune_perm_level.py
"""
from __future__ import annotations
import gc, os, sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.feature import default_mask, load_wells, make_features, make_targets
from src.utils import EPS, TARGETS

CURVES = ["GR", "PE", "SP", "CAL", "AC", "DEN", "CNL", "RXO", "RT", "DEVI", "AZIM", "BIT"]
EPS_ACC = 1e-5


def acc_perm_official(pred, true):
    """官方 PERM：mean(max(0, 1 - |log10(max(ŷ/y, ε))|))。恒 1.0 decade 比值容差。"""
    p = np.asarray(pred, np.float64); t = np.asarray(true, np.float64)
    return float(np.mean(np.maximum(0.0, 1.0 - np.abs(np.log10(np.maximum(p / np.maximum(t, EPS_ACC), EPS_ACC))))))


def contiguous_segments(lids, real_mask, min_len=3):
    out = []
    for lid in pd.unique(lids):
        ri = np.flatnonzero(real_mask & (lids == lid))
        for g in np.split(ri, np.flatnonzero(np.diff(ri) != 1) + 1):
            if len(g) >= min_len:
                out.append(g)
    return out


def seg_table(df, segs, por=None):
    """段级聚合特征。por 非空时追加段内 POR 均值/std。"""
    arr = df[CURVES].to_numpy(float)
    rows = []
    for g in segs:
        sub = arr[g]
        r = {"__len": len(g)}
        for j, col in enumerate(CURVES):
            v = sub[:, j]; ok = ~np.isnan(v)
            v = v[ok] if ok.any() else np.array([np.nan])
            r[f"{col}_m"] = np.mean(v); r[f"{col}_s"] = np.std(v)
            r[f"{col}_min"] = np.min(v); r[f"{col}_max"] = np.max(v)
        if por is not None:
            pv = por[g]
            r["POR_m"] = np.nanmean(pv); r["POR_s"] = np.nanstd(pv)
        rows.append(r)
    return pd.DataFrame(rows)


def main() -> None:
    t0 = time.time()
    wells = load_wells("./train", has_labels=True)
    ids = [w["logId"].iloc[0] for w in wells]
    tr_ids, va_ids = set(ids[:72]), set(ids[72:])
    tr_df = pd.concat([w for w in wells if w["logId"].iloc[0] in tr_ids], ignore_index=True)
    va_df = pd.concat([w for w in wells if w["logId"].iloc[0] in va_ids], ignore_index=True)
    del wells; gc.collect()

    trY, vaY = make_targets(tr_df), make_targets(va_df)
    trDM, vaDM = default_mask(trY), default_mask(vaY)
    ttr, tva = tr_df["logId"].to_numpy(), va_df["logId"].to_numpy()

    c = "PERM"
    tr_perm = trY[c].to_numpy(); va_perm = vaY[c].to_numpy()
    real_tr = (~trDM[c].to_numpy()) & ~np.isnan(tr_perm)
    real_va = (~vaDM[c].to_numpy()) & ~np.isnan(va_perm)
    ty_log = np.log10(np.maximum(tr_perm, EPS))
    vy_log = np.log10(np.maximum(va_perm, EPS))

    # POR：真值（上界）与树预测（可部署）。用同一批 72 井训练 POR 回归器。
    por_true_tr = trY["POR"].to_numpy()
    por_true_va = vaY["POR"].to_numpy()
    print("训练 POR 树（仅用于生成 POR 预测特征）...", flush=True)
    from src.model import _reg, POR_MONOTONIC, inverse_transform_target
    Xtr = make_features(tr_df, context=False, priors=True)
    Xva = make_features(va_df, context=False, priors=True)
    por_ok = trY["POR"].notna().to_numpy() & ~trDM["POR"].to_numpy()
    from src.feature import transform_target
    por_reg = _reg(monotonic=POR_MONOTONIC, columns=list(Xtr.columns))
    por_reg.fit(Xtr.loc[por_ok], transform_target("POR", trY.loc[por_ok, "POR"]).to_numpy())
    por_pred_tr = np.log10(np.maximum(por_true_tr, EPS))
    por_pred_tr[por_ok] = por_reg.predict(Xtr.loc[por_ok]).astype(np.float64)
    por_pred_va = por_reg.predict(Xva).astype(np.float64)
    del por_reg, Xtr, Xva; gc.collect()
    print(f"    POR 树完成  [{time.time()-t0:.0f}s]", flush=True)

    segs_tr = contiguous_segments(ttr, real_tr)
    segs_va = contiguous_segments(tva, real_va)
    lab_tr = np.array([np.nanmean(ty_log[g]) for g in segs_tr], float)
    lab_va = np.array([np.nanmean(vy_log[g]) for g in segs_va], float)

    # 逐点映射数组
    level_true = np.full(len(va_df), np.nan)
    for gi, g in enumerate(segs_va):
        level_true[g] = lab_va[gi]
    ev = real_va & ~np.isnan(level_true)

    a_oracle = acc_perm_official(10.0 ** level_true[ev], va_perm[ev])
    print(f"\n真值段均值 oracle（不可达上界）  PERM real acc = {a_oracle:.4f}   "
          f"[{time.time()-t0:.0f}s]")

    FEATSETS = [
        ("纯测井 48 维", None),
        ("+POR 真值（上界）", por_true_tr),
        ("+POR 树预测（可部署）", por_pred_tr),
    ]
    LOSSES = [
        ("squared_error", dict(loss="squared_error")),
        ("absolute_error", dict(loss="absolute_error")),
        ("quantile(0.5)", dict(loss="quantile", quantile=0.5)),
    ]
    HYPERS = [
        ("i400_lr0.06_leaf31", dict(max_iter=400, learning_rate=0.06, max_leaf_nodes=31,
                                    min_samples_leaf=20)),
        ("i800_lr0.04_leaf63", dict(max_iter=800, learning_rate=0.04, max_leaf_nodes=63,
                                    min_samples_leaf=15)),
        ("i400_lr0.1_leaf31", dict(max_iter=400, learning_rate=0.1, max_leaf_nodes=31,
                                   min_samples_leaf=20)),
    ]

    print(f"\n{'特征集':24s} {'loss':16s} {'超参':22s} {'PERM real acc':>13s} "
          f"{'对总分Δ(分)':>15s} {'≥75?':>6s}")
    print("-" * 100)
    best = None
    for name, por in FEATSETS:
        Ftr = seg_table(tr_df, segs_tr, por)
        Fva = seg_table(va_df, segs_va,
                        por_true_va if por is por_true_tr else
                        (por_pred_va if por is por_pred_tr else None))
        cols = [k for k in Ftr.columns if not k.startswith("__")]
        ok = ~np.isnan(lab_tr)
        for lname, kw in LOSSES:
            for hname, hk in HYPERS:
                m = HistGradientBoostingRegressor(random_state=0, **kw, **hk)
                m.fit(Ftr.loc[ok, cols].to_numpy(float), lab_tr[ok])
                lv = m.predict(Fva[cols].to_numpy(float))
                # 逐点映射后再取 mean —— 必须逐点等权（按段等权会系统性虚高 +2.3 量级）
                lvl_pt = np.full(len(va_df), np.nan)
                for gi in range(len(segs_va)):
                    lvl_pt[segs_va[gi]] = lv[gi]
                a = float(acc_perm_official(10.0 ** lvl_pt[ev], va_perm[ev]))
                ok75 = "✓" if a >= 0.5241 else ""
                print(f"{name:24s} {lname:16s} {hname:22s} {a:13.4f} "
                      f"{(a-0.4015)*0.35*0.3937*100:+15.3f}分 {ok75:>6s}")
                if best is None or a > best[0]:
                    best = (a, name, lname, hname)
                del m, lv, lvl_pt; gc.collect()
        del Ftr, Fva; gc.collect()

    print("-" * 100)
    print(f"最佳: {best[1]} + {best[2]} + {best[3]}  → PERM real {best[0]:.4f}")
    print(f"      对总分的影响: {(best[0]-0.4015) * 0.35 * 0.3937 * 100:+.3f} 分 "
          f"(0.35 权重 × (1-f_def=0.3937))")
    print(f"\n注: 全部为真值分段上界。可部署性另测（分类器分段会损失一点）。")


if __name__ == "__main__":
    main()
