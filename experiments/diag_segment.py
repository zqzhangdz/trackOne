"""真实段误差结构诊断：段级水平 vs 点级偏差；段均值 oracle 上界；段聚合特征相关性."""
from __future__ import annotations
import sys, os, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.feature import load_wells, make_features, make_targets, default_mask, transform_target
from src.utils import TARGETS, TOLERANCE, EPS

def pw(c, p, t):
    p = np.asarray(p, float); t = np.asarray(t, float)
    if c == "PERM":
        p = np.log10(np.maximum(p, EPS)); t = np.log10(np.maximum(t, EPS))
    return np.maximum(0.0, 1.0 - np.abs(p - t) / (TOLERANCE[c] * (np.abs(t) + EPS)))

def segments(mask: np.ndarray):
    """返回极大连续 True 区间列表."""
    idx = np.flatnonzero(np.diff(np.concatenate([[0], mask.view(np.int8), [0]])))
    return list(zip(idx[::2], idx[1::2]))

def main():
    wells = load_wells("./train", has_labels=True)
    stats = {c: {"var_between": [], "var_within": [], "seg_len": [],
                 "acc_segmean": [], "acc_segmean_causal": [], "corr_seg": []} for c in TARGETS}
    for w in wells:
        Y = make_targets(w); isd = default_mask(Y)
        X = make_features(w, context=False, priors=True)
        for c in TARGETS:
            y = Y[c].to_numpy(); valid = ~np.isnan(y)
            real = valid & ~isd[c].to_numpy()
            t = transform_target(c, pd.Series(np.where(valid, y, np.nan))).to_numpy()
            for (s, e) in segments(real):
                if e - s < 3:
                    continue
                seg_t = t[s:e]
                m = np.mean(seg_t)
                stats[c]["seg_len"].append(e - s)
                stats[c]["var_between"].append(m ** 2 * (e - s))
                stats[c]["var_within"].append(np.sum((seg_t - m) ** 2))
                # oracle: 段均值(全段, 非因果)
                stats[c]["acc_segmean"].append(pw(c, np.full(e - s, m), y[s:e]).mean() * (e - s))
                # oracle: 因果段均值(仅用段内已过的点, 模拟追踪)
                cm = np.cumsum(seg_t) / np.arange(1, e - s + 1)
                stats[c]["acc_segmean_causal"].append(pw(c, cm, y[s:e]).mean() * (e - s))
                # 段聚合特征与段均值的相关性样本
                agg = X.iloc[s:e].mean(numeric_only=True)
                stats[c]["corr_seg"].append((agg, m, e - s))
    for c in TARGETS:
        vb = sum(stats[c]["var_between"]); vw = sum(stats[c]["var_within"])
        n = sum(stats[c]["seg_len"])
        acc_sm = sum(stats[c]["acc_segmean"]) / n
        acc_cm = sum(stats[c]["acc_segmean_causal"]) / n
        print(f"{c}: 段数={len(stats[c]['seg_len'])} 平均段长={np.mean(stats[c]['seg_len']):.1f} "
              f"方差占比 between={vb / (vb + vw):.3f} within={vw / (vb + vw):.3f}")
        print(f"   oracle 段均值(全段)={acc_sm:.4f}  因果段均值={acc_cm:.4f}")
        # 段聚合特征 vs 段均值 相关系数
        df = pd.DataFrame([a for a, _m, _n in stats[c]["corr_seg"]])
        mv = np.array([_m for _a, _m, _n in stats[c]["corr_seg"]])
        wts = np.array([_n for _a, _m, _n in stats[c]["corr_seg"]])
        cors = []
        for col in ["DEN", "AC", "CNL", "GR", "logRT", "POR_D", "POR_avg"]:
            if col in df.columns:
                r = np.corrcoef(df[col].to_numpy(), mv)[0, 1]
                cors.append(f"{col}={r:+.2f}")
        print("   段均值 vs 段聚合特征相关:", " ".join(cors[:7]))

if __name__ == "__main__":
    main()
