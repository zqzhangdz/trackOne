"""A) 逐井迁移: 仅用 NN 训练井真实段建模 vs 全局模型; B) 段级 kNN 非参 level."""
from __future__ import annotations
import sys, os, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.feature import (default_mask, inverse_transform_target, load_wells, make_features,
                         make_targets, transform_target)
from src.model import TwoStageModel
from src.utils import EPS, TARGETS, TOLERANCE

def pw(c, p, t):
    p = np.asarray(p, float); t = np.asarray(t, float)
    if c == "PERM":
        p = np.log10(np.maximum(p, EPS)); t = np.log10(np.maximum(t, EPS))
    return np.maximum(0.0, 1.0 - np.abs(p - t) / (TOLERANCE[c] * (np.abs(t) + EPS)))

def segments_of(mask):
    idx = np.flatnonzero(np.diff(np.concatenate([[0], mask.view(np.int8), [0]])))
    return list(zip(idx[::2], idx[1::2]))

LOGCOLS = ["GR", "PE", "SP", "CAL", "AC", "DEN", "CNL", "logRT", "logRXO", "POR_D", "POR_S", "POR_avg"]

def main():
    wells = load_wells("./train", has_labels=True)
    ids = [w["logId"].iloc[0] for w in wells]
    tr_ids = ids[:72]
    va = [w for w in wells if w["logId"].iloc[0] in set(ids[72:])]
    tr = [w for w in wells if w["logId"].iloc[0] in set(tr_ids)]
    tr_df = pd.concat(tr, ignore_index=True)

    # 全局模型 (screen 训练集)
    Xtr = make_features(tr_df, context=True, priors=True)
    Ytr = make_targets(tr_df)
    glob = TwoStageModel(monotonic=True).fit(Xtr, Ytr, default_mask(Ytr))

    # 井指纹
    def fp(w):
        X = make_features(w, context=False, priors=False)
        v = []
        for c in LOGCOLS[:9]:
            v += [float(X[c].mean()), float(X[c].std())]
        return np.array(v)
    F_tr = np.stack([fp(w) for w in tr])
    mu, sd = F_tr.mean(0), F_tr.std(0) + 1e-9
    Ztr = (F_tr - mu) / sd

    # 段级指纹库 (训练井真段)
    seg_lib = {c: {"F": [], "lvl": [], "wid": []} for c in TARGETS}
    for wi, w in enumerate(tr):
        Yw = make_targets(w); isdw = default_mask(Yw)
        Xw = make_features(w, context=False, priors=True)
        for c in TARGETS:
            y = Yw[c].to_numpy(); valid = ~np.isnan(y)
            real = valid & ~isdw[c].to_numpy()
            t = transform_target(c, pd.Series(np.where(valid, y, np.nan))).to_numpy()
            for (s, e) in segments_of(real):
                if e - s < 5:
                    continue
                g = Xw.iloc[s:e][LOGCOLS]
                f = np.concatenate([g.mean().to_numpy(), g.std().fillna(0).to_numpy(), [np.log(e - s)]])
                seg_lib[c]["F"].append(f)
                seg_lib[c]["lvl"].append(float(np.nanmean(t[s:e])))
                seg_lib[c]["wid"].append(wi)
    for c in TARGETS:
        Fm = np.stack(seg_lib[c]["F"])
        seg_lib[c]["Fz"] = (Fm - Fm.mean(0)) / (Fm.std(0) + 1e-9)
        seg_lib[c]["lvl"] = np.array(seg_lib[c]["lvl"])

    print("=== A) 逐井迁移 vs 全局 (验证井真实段, 树回归) ===")
    acc_g, acc_l = {c: [] for c in TARGETS}, {c: [] for c in TARGETS}
    for w in va:
        z = (fp(w) - mu) / sd
        nn = int(np.argmin(np.sqrt(((Ztr - z) ** 2).mean(-1))))
        lw = tr[nn]
        Xlw = make_features(lw, context=True, priors=True)
        Ylw = make_targets(lw)
        local = TwoStageModel(monotonic=True).fit(Xlw, Ylw, default_mask(Ylw))
        Xw = make_features(w, context=True, priors=True)
        Yw = make_targets(w); isdw = default_mask(Yw)
        for c in TARGETS:
            y = Yw[c].to_numpy(); valid = ~np.isnan(y)
            real = valid & ~isdw[c].to_numpy()
            if real.sum() < 50:
                continue
            pg = inverse_transform_target(c, pd.Series(glob.regressors[c].predict(Xw), index=Xw.index)).to_numpy()
            pl = inverse_transform_target(c, pd.Series(local.regressors[c].predict(Xw), index=Xw.index)).to_numpy()
            acc_g[c].append(pw(c, pg[real], y[real]).mean())
            acc_l[c].append(pw(c, pl[real], y[real]).mean())
    for c in TARGETS:
        print(f"{c}: 全局={np.mean(acc_g[c]):.4f}  仅NN井={np.mean(acc_l[c]):.4f}  delta={np.mean(acc_l[c]) - np.mean(acc_g[c]):+.4f}")

    print("=== B) 段级 kNN level (验证井真段) ===")
    for c in TARGETS:
        accs = {k: [] for k in (5, 15, 40)}
        for w in va:
            Yw = make_targets(w); isdw = default_mask(Yw)
            Xw = make_features(w, context=False, priors=True)
            y = Yw[c].to_numpy(); valid = ~np.isnan(y)
            real = valid & ~isdw[c].to_numpy()
            t = transform_target(c, pd.Series(np.where(valid, y, np.nan))).to_numpy()
            for (s, e) in segments_of(real):
                if e - s < 5:
                    continue
                g = Xw.iloc[s:e][LOGCOLS]
                f = np.concatenate([g.mean().to_numpy(), g.std().fillna(0).to_numpy(), [np.log(e - s)]])
                Fm = np.stack(seg_lib[c]["F"])
                fz = (f - Fm.mean(0)) / (Fm.std(0) + 1e-9)
                d = np.sqrt(((seg_lib[c]["Fz"] - fz) ** 2).mean(-1))
                pred_phys = {}
                for k in accs:
                    idx = np.argsort(d)[:k]
                    lvl = seg_lib[c]["lvl"][idx].mean()
                    pred_phys[k] = np.full(e - s, float(inverse_transform_target(c, pd.Series([lvl])).iloc[0]))
                for k in accs:
                    accs[k].append(pw(c, pred_phys[k], y[s:e]).mean() * (e - s))
        n = None
        line = []
        for k in accs:
            nseg_pts = sum(1 for w in va for (s, e) in segments_of((~np.isnan(make_targets(w)[c].to_numpy())) & ~default_mask(make_targets(w))[c].to_numpy()) if e - s >= 5 for _ in range(e - s))
            line.append(f"k={k}: {sum(accs[k]) / nseg_pts:.4f}")
        print(f"{c}: " + "  ".join(line))

if __name__ == "__main__":
    main()
