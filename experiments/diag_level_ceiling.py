"""测量「段水平估计」的现实天花板（不用真值分段，只用测井聚合特征）。

背景：diag_oracle_seg.py 显示 PERM 真实段 acc，模型 0.073 vs 真值段均值 oracle 0.311，
是全项目最大单点缺口（约占总缺口 70%）。本脚本回答：
不允许用真值时，只用测井曲线聚合特征估段水平，各目标的现实可达 acc 是多少？

口径说明（诚实标注）：
  - 分段用【真值分段】划分段边界（仅用于构造训练标签），评估逐点计算。
    故本结果是「已知分段时，水平估计能做到的上界」，不含分段误差。
  - 每个目标各自分段，段内 log10(y) 均值为标签；评估在真值真实段上逐点计算。

纯 HistGB，不涉及 torch；安全于同进程运行时约束。
用法: python experiments/diag_level_ceiling.py
"""
from __future__ import annotations
import os, sys, warnings, time
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.feature import default_mask, load_wells, make_targets
from src.utils import DEFAULT_VALUES, EPS, TARGETS, TOLERANCE, WEIGHTS

CURVES = ["GR", "PE", "SP", "CAL", "AC", "DEN", "CNL", "RXO", "RT", "DEVI", "AZIM", "BIT"]


def pw(c, pred, true):
    """截断相对误差准确率（逐点）。pred/true 同为物理空间。"""
    p = np.asarray(pred, float); t = np.asarray(true, float)
    assert p.shape == t.shape, f"shape 不匹配 {p.shape} vs {t.shape}"
    if c == "PERM":
        p = np.log10(np.maximum(p, EPS)); t = np.log10(np.maximum(t, EPS))
    return float(np.mean(np.maximum(0.0, 1.0 - np.abs(p - t) / (TOLERANCE[c] * (np.abs(t) + EPS)))))


def contiguous_segments(lids, real_mask, min_len=3):
    """按连续深度返回每个段内的行索引数组列表。必须遍历唯一井号。"""
    out = []
    for lid in pd.unique(lids):
        ri = np.flatnonzero(real_mask & (lids == lid))
        if ri.size == 0:
            continue
        for g in np.split(ri, np.flatnonzero(np.diff(ri) != 1) + 1):
            if len(g) >= min_len:
                out.append(g)
    return out


def seg_feature_table(df, segs):
    """段级聚合特征：各曲线 mean/std/min/max。"""
    arr = df[CURVES].to_numpy(float)
    rows = []
    for g in segs:
        sub = arr[g]
        r = {"__len": len(g)}
        for j, col in enumerate(CURVES):
            v = sub[:, j]
            ok = ~np.isnan(v)
            v = v[ok] if ok.any() else np.array([np.nan])
            r[f"{col}_m"] = np.mean(v); r[f"{col}_s"] = np.std(v)
            r[f"{col}_min"] = np.min(v); r[f"{col}_max"] = np.max(v)
        rows.append(r)
    return pd.DataFrame(rows)


def main() -> None:
    wells = load_wells("./train", has_labels=True)
    ids = [w["logId"].iloc[0] for w in wells]
    tr_ids, va_ids = ids[:72], ids[72:]
    tr_df = pd.concat([w for w in wells if w["logId"].iloc[0] in set(tr_ids)], ignore_index=True)
    va_df = pd.concat([w for w in wells if w["logId"].iloc[0] in set(va_ids)], ignore_index=True)
    trY, vaY = make_targets(tr_df), make_targets(va_df)
    trDM, vaDM = default_mask(trY), default_mask(vaY)
    ttr = tr_df["logId"].to_numpy(); tva = va_df["logId"].to_numpy()
    t0 = time.time()

    print(f"{'目标':6s} {'真值段均值oracle':>15s} {'HistGB估水平':>13s} {'差':>9s} "
          f"{'HistGB→oracle 保留比':>20s} {'段内log10-RMSE':>16s} {'段数':>6s}")
    for c in TARGETS:
        ty_arr = np.log10(np.maximum(trY[c].to_numpy(), EPS))
        vy_arr = np.log10(np.maximum(vaY[c].to_numpy(), EPS))
        real_tr = (~trDM[c].to_numpy()) & ~np.isnan(ty_arr)
        real_va = (~vaDM[c].to_numpy()) & ~np.isnan(vy_arr)

        segs_tr = contiguous_segments(ttr, real_tr)
        segs_va = contiguous_segments(tva, real_va)
        Ftr = seg_feature_table(tr_df, segs_tr)
        Fva = seg_feature_table(va_df, segs_va)
        cols = [k for k in Ftr.columns if k.startswith(tuple(f"{x}_" for x in CURVES))]

        # 标签：段内 log10(y) 均值
        def seg_means(segs, ylog):
            out = []
            for g in segs:
                v = ylog[g]; out.append(np.nanmean(v))
            return np.asarray(out, float)
        lab_tr, lab_va = seg_means(segs_tr, ty_arr), seg_means(segs_va, vy_arr)

        ok = ~np.isnan(lab_tr)
        mdl = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06,
                                            max_leaf_nodes=31, min_samples_leaf=20, random_state=0)
        mdl.fit(Ftr.loc[ok, cols].to_numpy(float), lab_tr[ok])
        lv = mdl.predict(Fva[cols].to_numpy(float))

        # 逐点映射
        level_pred = np.full(len(va_df), np.nan)
        segmean_true = np.full(len(va_df), np.nan)
        for gi, g in enumerate(segs_va):
            level_pred[g] = lv[gi]
            segmean_true[g] = lab_va[gi]

        ev = real_va & ~np.isnan(level_pred)
        a_oracle = pw(c, np.power(10.0, segmean_true[ev]), vaY[c].to_numpy()[ev])
        a_model = pw(c, np.power(10.0, level_pred[ev]), vaY[c].to_numpy()[ev])
        rmse = float(np.sqrt(np.nanmean((lv - lab_va) ** 2)))
        ratio = (a_model / a_oracle) if a_oracle > 0 else float("nan")
        print(f"{c:6s} {a_oracle:15.4f} {a_model:13.4f} {a_model-a_oracle:9.4f} "
              f"{ratio:20.3f} {rmse:16.4f} {len(segs_va):6d}   [{time.time()-t0:.0f}s]")

    print("\n注: oracle=真值段均值(上界)；HistGB=仅用测井聚合特征估水平（已含分段误差前）。")


if __name__ == "__main__":
    main()
