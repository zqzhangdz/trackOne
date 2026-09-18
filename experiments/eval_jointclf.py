"""联合分段分类实验（screen 验证井）.

结构观察：三目标默认段 Jaccard>0.85，存在强共享的“储层区域”结构。
测试用区域级分类器（all-default / any-real）修正逐目标 p_def：
  (a) 逐目标 clf（基线）
  (b) 区域 clf 直接用
  (c) 几何/算术混合 p_def = f(p_target, p_region)
回归固定为 screen_tree_cache 的 tree 回归（隔离分类效应）。
"""
from __future__ import annotations
import os, sys, warnings
warnings.filterwarnings("ignore")
import joblib, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.feature import default_mask, load_wells, make_features, make_targets
from src.model import _clf
from src.utils import DEFAULT_VALUES, EPS, TARGETS, TOLERANCE, WEIGHTS

def pointwise(c, pred, true):
    p = np.asarray(pred, float); t = np.asarray(true, float)
    if c == "PERM":
        p = np.log10(np.maximum(p, EPS)); t = np.log10(np.maximum(t, EPS))
    return np.maximum(0.0, 1.0 - np.abs(p - t) / (TOLERANCE[c] * (np.abs(t) + EPS)))

def main():
    wells = load_wells("./train", has_labels=True)
    ids = [w["logId"].iloc[0] for w in wells]
    tr = [w for w in wells if w["logId"].iloc[0] in set(ids[:72])]
    va = [w for w in wells if w["logId"].iloc[0] in set(ids[72:])]
    tr_df, va_df = pd.concat(tr, ignore_index=True), pd.concat(va, ignore_index=True)
    Xtr = make_features(tr_df, context=True, priors=True)
    Xva = make_features(va_df, context=True, priors=True)
    Ytr, Yva = make_targets(tr_df), make_targets(va_df)
    dtr = default_mask(Ytr); dva = default_mask(Yva)
    cache = joblib.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "screen_tree_cache.joblib"))
    reg = cache["reg"]; p_tree = cache["p_def"]

    # 训练标签（默认段为 1）
    y_all_def = (dtr["POR"] & dtr["PERM"] & dtr["SW"]).astype(int).to_numpy()
    y_any_real = (~(dtr["POR"] | dtr["PERM"] | dtr["SW"])).astype(int).to_numpy()
    m_all_def = _clf(); m_all_def.fit(Xtr, y_all_def)
    m_any_real = _clf(); m_any_real.fit(Xtr, y_any_real)
    p_all = m_all_def.predict_proba(Xva)[:, 1]
    p_any = m_any_real.predict_proba(Xva)[:, 1]

    # 区域标签的任务版本（truth）
    da = (dva["POR"] & dva["PERM"] & dva["SW"]).to_numpy()
    dr = (~(dva["POR"] | dva["PERM"] | dva["SW"])).to_numpy()
    print(f"区域一致性: all-default 点={da.sum()} any-real 点={dr.sum()} | 不一致={len(va_df)-da.sum()-dr.sum()}")

    grid = np.arange(0.02, 0.71, 0.02)
    def ev(pm, name):
        tot = 0.0; det = []
        for c in TARGETS:
            y = Yva[c].to_numpy(); m = ~np.isnan(y)
            bt, ba = 0.5, -1
            for t in grid:
                pred = np.where(pm[c] >= t, DEFAULT_VALUES[c], np.power(10.0, reg[c]))
                a = pointwise(c, pred[m], y[m]).mean()
                if a > ba: ba, bt = a, float(t)
            tot += WEIGHTS[c]*ba; det.append(f"{c}:{ba:.4f}@{bt:.2f}")
        print(f"{name:28s}: {tot*100:.4f}  " + " ".join(det))
        return tot*100

    ev(p_tree, "per-target clf (baseline)")
    ev({c: p_all for c in TARGETS}, "region all-default only")
    ev({c: 0.5*p_tree[c]+0.5*p_all for c in TARGETS}, "arith .5 target+.5 all")
    for a in (0.2, 0.3, 0.4, 0.5):
        ev({c: np.clip(p_tree[c],1e-6,1)**1 * (p_all**a) for c in TARGETS}, f"geom target*all^{a}")
    for a in (0.2, 0.3, 0.4):
        ev({c: p_tree[c] * (p_all**a) / np.clip((p_any**a),1e-6,1) for c in TARGETS}, f"geom target*(all/any)^{a}")
    np.savez(os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "jointclf_val.npz"),
             p_all=p_all, p_any=p_any, da=da, dr=dr)

if __name__ == "__main__":
    main()
