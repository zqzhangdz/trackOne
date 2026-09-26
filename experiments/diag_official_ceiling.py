"""官方口径下的天花板重算（2026-09-26 修正 PERM 公式后）。

为什么重算：`diag_level_ceiling.py` 的分母是旧 PERM 容差 0.08·|log10 y|，
把 PERM 的真值段均值 oracle 低估成 0.3110（实际官方口径高得多）。
本脚本用官方 PERM = mean(max(0, 1 − |log10(max(ŷ/y, ε))|)) 重算整张天花板表，
并给出「要到 75 分，每个目标 real acc 需要到多少」的敏感度。

口径（与 diag_level_ceiling.py 完全一致，只有评分函数变了）：
  - 分段用【真值分段】划边界（仅用于构造训练标签），评估逐点计算。
    → 本结果是「已知分段时」的上界，不含分段误差。
  - 每个目标各自分段；段内 log10(y) 均值为标签。

内存：仅 HistGB（段级样本数 ~数千），无 torch，无大矩阵。峰值 < 1 GB。
用法: python experiments/diag_official_ceiling.py
"""
from __future__ import annotations
import os, sys, gc, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.feature import default_mask, load_wells, make_targets
from src.utils import DEFAULT_VALUES, EPS, TARGETS, WEIGHTS

CURVES = ["GR", "PE", "SP", "CAL", "AC", "DEN", "CNL", "RXO", "RT", "DEVI", "AZIM", "BIT"]


def acc_pointwise(c, pred, true):
    """官方口径逐点准确率向量（不取 mean）。PERM 恒 1.0 decade 比值容差。"""
    p = np.asarray(pred, np.float64)
    t = np.asarray(true, np.float64)
    if c == "PERM":
        return np.maximum(0.0, 1.0 - np.abs(np.log10(np.maximum(p / np.maximum(t, EPS), EPS))))
    tol = 0.08 if c == "POR" else 0.05
    return np.maximum(0.0, 1.0 - np.abs(p - t) / (tol * (np.abs(t) + EPS)))


def contiguous_segments(lids, real_mask, min_len=3):
    out = []
    for lid in pd.unique(lids):
        ri = np.flatnonzero(real_mask & (lids == lid))
        for g in np.split(ri, np.flatnonzero(np.diff(ri) != 1) + 1):
            if len(g) >= min_len:
                out.append(g)
    return out


def seg_feature_table(df, segs):
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
    t0 = time.time()
    wells = load_wells("./train", has_labels=True)
    ids = [w["logId"].iloc[0] for w in wells]
    tr_ids, va_ids = ids[:72], ids[72:]
    tr_df = pd.concat([w for w in wells if w["logId"].iloc[0] in set(tr_ids)], ignore_index=True)
    va_df = pd.concat([w for w in wells if w["logId"].iloc[0] in set(va_ids)], ignore_index=True)
    trY, vaY = make_targets(tr_df), make_targets(va_df)
    trDM, vaDM = default_mask(trY), default_mask(vaY)
    ttr, tva = tr_df["logId"].to_numpy(), va_df["logId"].to_numpy()
    gc.collect()

    acc_def = {}
    f_def = {}
    rows = {}
    for c in TARGETS:
        yva = vaY[c].to_numpy()
        d = vaDM[c].to_numpy()
        m = ~np.isnan(yva)
        acc_def[c] = float(acc_pointwise(c, np.full(len(yva), DEFAULT_VALUES[c]), yva)[m & d].mean())
        f_def[c] = float((m & d).sum() / m.sum())
        rows[c] = {"acc_def": acc_def[c], "f_def": f_def[c]}

    print(f"{'目标':6s} {'默认段占比':>9s} {'acc_def':>8s} {'oracle真段均值':>13s} "
          f"{'HistGB估水平':>12s} {'HistGB→oracle':>14s} {'段内log10-RMSE':>16s} {'段数':>6s}")
    for c in TARGETS:
        ylog = np.log10(np.maximum(trY[c].to_numpy(), EPS))
        vlog = np.log10(np.maximum(vaY[c].to_numpy(), EPS))
        real_tr = (~trDM[c].to_numpy()) & ~np.isnan(ylog)
        real_va = (~vaDM[c].to_numpy()) & ~np.isnan(vlog)

        segs_tr = contiguous_segments(ttr, real_tr)
        segs_va = contiguous_segments(tva, real_va)
        Ftr = seg_feature_table(tr_df, segs_tr)
        Fva = seg_feature_table(va_df, segs_va)
        cols = [k for k in Ftr.columns if k.startswith(tuple(f"{x}_" for x in CURVES))]
        gc.collect()

        lab_tr = np.array([np.nanmean(ylog[g]) for g in segs_tr], float)
        lab_va = np.array([np.nanmean(vlog[g]) for g in segs_va], float)
        ok = ~np.isnan(lab_tr)
        mdl = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06,
                                            max_leaf_nodes=31, min_samples_leaf=20, random_state=0)
        mdl.fit(Ftr.loc[ok, cols].to_numpy(float), lab_tr[ok])
        lv = mdl.predict(Fva[cols].to_numpy(float))
        del mdl
        gc.collect()

        level_pred = np.full(len(va_df), np.nan)
        segmean_true = np.full(len(va_df), np.nan)
        for gi, g in enumerate(segs_va):
            level_pred[g] = lv[gi]
            segmean_true[g] = lab_va[gi]

        ev = real_va & ~np.isnan(level_pred)
        yv = vaY[c].to_numpy()
        a_or = float(acc_pointwise(c, 10.0 ** segmean_true[ev], yv[ev]).mean())
        a_lv = float(acc_pointwise(c, 10.0 ** level_pred[ev], yv[ev]).mean())
        rmse = float(np.sqrt(np.nanmean((lv - lab_va) ** 2)))
        rows[c]["oracle"] = a_or
        rows[c]["level"] = a_lv
        print(f"{c:6s} {f_def[c]:9.4f} {acc_def[c]:8.4f} {a_or:13.4f} {a_lv:12.4f} "
              f"{(a_lv/a_or if a_or > 0 else float('nan')):14.3f} {rmse:16.4f} {len(segs_va):6d} "
              f"   [{time.time()-t0:.0f}s]")
        gc.collect()

    # ── 总分口径：acc = f_def·acc_def + (1-f_def)·acc_real
    def total(real):
        a = {c: f_def[c] * acc_def[c] + (1 - f_def[c]) * real[c] for c in TARGETS}
        return float(sum(WEIGHTS[c] * a[c] for c in TARGETS) * 100), a

    print("\n" + "=" * 96)
    print("官方口径总分（screen 8 井，acc_def 用实测 v6 值，POR/SW real 用 v6 实测）")
    print("=" * 96)
    print(f"{'配置':34s} {'POR_real':>9s} {'PERM_real':>9s} {'SW_real':>9s} {'Total':>8s}")

    V6_REAL = {"POR": 0.1694, "PERM": 0.4015, "SW": 0.1450}
    t_v6, _ = total(V6_REAL)
    print(f"{'v6 当前（交付工件实测）':32s} {V6_REAL['POR']:9.4f} {V6_REAL['PERM']:9.4f} "
          f"{V6_REAL['SW']:9.4f} {t_v6:8.3f}")

    for c in TARGETS:
        r = dict(V6_REAL); r[c] = rows[c]["oracle"]
        t, _ = total(r)
        print(f"{'v6 + '+c+'→真值段均值oracle':32s} {r['POR']:9.4f} {r['PERM']:9.4f} "
              f"{r['SW']:9.4f} {t:8.3f}")
    for c in TARGETS:
        r = dict(V6_REAL); r[c] = rows[c]["level"]
        t, _ = total(r)
        print(f"{'v6 + '+c+'→HistGB段水平':32s} {r['POR']:9.4f} {r['PERM']:9.4f} "
              f"{r['SW']:9.4f} {t:8.3f}")

    r = {c: rows[c]["oracle"] for c in TARGETS}
    t, _ = total(r)
    print(f"{'三目标全 oracle（不可达上界）':29s} {r['POR']:9.4f} {r['PERM']:9.4f} "
          f"{r['SW']:9.4f} {t:8.3f}")

    print("\n" + "=" * 96)
    print("要到 75 分：在 v6 现状下，每个目标 real acc 单独提到多少可够（其余不动）")
    print("=" * 96)
    for c in TARGETS:
        d = WEIGHTS[c] * (1 - f_def[c]) * 100.0
        need = (75.0 - t_v6) / d
        r = dict(V6_REAL); r[c] = V6_REAL[c] + need
        print(f"  {c:5s} real: {V6_REAL[c]:.4f} → {r[c]:.4f}  (Δ{need:+.4f}, "
              f"oracle 是 {rows[c]['oracle']:.4f}, "
              f"{'已超 oracle' if r[c] > rows[c]['oracle'] else '可达'})")

    print("\n注: oracle = 真值段均值（需要真值分段边界，推理时不可得）；"
          "HistGB = 仅测井聚合特征估段水平（可用）。")


if __name__ == "__main__":
    main()
