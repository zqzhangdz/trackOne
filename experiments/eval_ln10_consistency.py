"""LN10 损失修正的严谨对照 + 训练/部署一致性诊断（screen 验证井，纯 numpy）.

背景: experiments/run_tcn.py 的 LN10 常量 bug(log10(e)*log(10)=1.0) 使 POR/SW 回归
损失被低估 2.3x；train.py 一直用正确 ln(10)。因此 screen 调优工件(T6/T8/T9, buggy)
与部署工件(train.py, corrected) 不同源。本脚本对比两族工件在 v4 融合协议下的表现，
并用逐井 bootstrap 给出差异置信区间。
"""
from __future__ import annotations
import os, sys, warnings
warnings.filterwarnings("ignore")
import joblib, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.feature import default_mask, load_wells, make_features, make_targets
from src.utils import DEFAULT_VALUES, EPS, TARGETS, TOLERANCE, WEIGHTS
from solution.tcn_model import numpy_forward

RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
BUGGY = ["T6_long", "T8_seed43"]
CORR = ["T10_ln10", "T11_seed43_ln10"]
V4_W = {"POR": 0.60, "PERM": 0.95, "SW": 0.75}


def pw(c, pred, true):
    p = np.asarray(pred, float); t = np.asarray(true, float)
    if c == "PERM":
        p = np.log10(np.maximum(p, EPS)); t = np.log10(np.maximum(t, EPS))
    return np.maximum(0.0, 1.0 - np.abs(p - t) / (TOLERANCE[c] * (np.abs(t) + EPS)))


def main():
    wells = load_wells("./train", has_labels=True); ids = [w["logId"].iloc[0] for w in wells]
    va = [w for w in wells if w["logId"].iloc[0] in set(ids[72:])]
    va_df = pd.concat(va, ignore_index=True); Yva = make_targets(va_df); tdef = default_mask(Yva)
    larr = va_df["logId"].to_numpy()
    cache = joblib.load(os.path.join(RUNS, "screen_tree_cache.joblib"))
    reg_tree, p_tree = cache["reg"], cache["p_def"]

    regs, pdfs = {}, {}
    for tag in BUGGY + CORR + ["T9_seed44"]:
        fp = os.path.join(RUNS, f"tcn_{tag}.joblib")
        if not os.path.exists(fp): continue
        art = joblib.load(fp); r, pp = {}, {}
        for w in va:
            Xw = make_features(w, context=False, priors=True).fillna(art["medians"])
            Xw = ((Xw - art["stds"].replace(0, 1.0)) / art["stds"].replace(0, 1.0)).to_numpy(np.float32)
            o = numpy_forward(Xw, art["sd"], art["dilations"])
            for c in TARGETS:
                r.setdefault(c, []).append(o[c][:, 1]); pp.setdefault(c, []).append(1/(1+np.exp(-o[c][:,0])))
        regs[tag] = {c: np.concatenate(v) for c, v in r.items()}
        pdfs[tag] = {c: np.concatenate(v) for c, v in pp.items()}

    print("=== 1) 纯回归（无默认决策，threshold=0）真实段 acc ===")
    for tag in regs:
        row = []
        for c in TARGETS:
            y = Yva[c].to_numpy(); m = (~np.isnan(y)) & (~tdef[c].to_numpy())
            row.append(pw(c, np.power(10., regs[tag][c])[m], y[m]).mean())
        print(f"  {tag:22s} real_reg_only=({row[0]:.4f},{row[1]:.4f},{row[2]:.4f})  mean={np.mean(row):.4f}")

    grid = np.arange(.02, .71, .02); wg = np.arange(0.0, 1.01, .05)

    def tuned_total(tags, clf_w=0.5, forced_w=None):
        tot, th, ws, parts = 0.0, {}, {}, {}
        for c in TARGETS:
            ens = np.mean([regs[t][c] for t in tags], axis=0)
            pens = np.mean([pdfs[t][c] for t in tags], axis=0)
            y = Yva[c].to_numpy(); m = ~np.isnan(y)
            pm = (1 - clf_w) * p_tree[c] + clf_w * pens
            if forced_w is None:
                ba, bt, bw = -1, .5, .7
                for w in wg:
                    for t in grid:
                        pred = np.where(pm >= t, DEFAULT_VALUES[c], np.power(10., (1-w)*reg_tree[c]+w*ens))
                        a = pw(c, pred[m], y[m]).mean()
                        if a > ba: ba, bt, bw = a, float(t), float(w)
            else:
                bw = forced_w[c]; ba, bt = -1, .5
                for t in grid:
                    pred = np.where(pm >= t, DEFAULT_VALUES[c], np.power(10., (1-bw)*reg_tree[c]+bw*ens))
                    a = pw(c, pred[m], y[m]).mean()
                    if a > ba: ba, bt = a, float(t)
            th[c], ws[c] = bt, bw
            pred = np.where(pm >= bt, DEFAULT_VALUES[c], np.power(10., (1-bw)*reg_tree[c]+bw*ens))
            parts[c] = (pw(c, pred[m], y[m]).mean(),
                        pw(c, pred[m & ~tdef[c].to_numpy()], y[m & ~tdef[c].to_numpy()]).mean())
            tot += WEIGHTS[c] * ba
        return tot * 100, th, ws, parts

    print("\n=== 2) v4 融合协议（clf_w=0.5），逐目标调优 vs 固定 v4 权重 ===")
    for name, tags in [("buggy(T6+T8, v4 screen 源)", BUGGY), ("corrected(T10+T11, 部署同源)", CORR),
                       ("buggy+corrected 4路", BUGGY + CORR)]:
        t1, th1, w1, p1 = tuned_total(tags)
        print(f"  {name:28s} tuned  : total={t1:.3f} real=({p1['POR'][1]:.4f},{p1['PERM'][1]:.4f},{p1['SW'][1]:.4f}) w=({w1['POR']:.2f},{w1['PERM']:.2f},{w1['SW']:.2f})")
        t2, th2, _, p2 = tuned_total(tags, forced_w=V4_W)
        print(f"  {name:28s} v4固定w: total={t2:.3f} real=({p2['POR'][1]:.4f},{p2['PERM'][1]:.4f},{p2['SW'][1]:.4f})")

    print("\n=== 3) 逐目标 w_tcn 曲线（clf_w=0.5, threshold 调优）===")
    for c in TARGETS:
        line = []
        for name, tags in [("buggy", BUGGY), ("corr", CORR)]:
            ens = np.mean([regs[t][c] for t in tags], axis=0)
            pens = np.mean([pdfs[t][c] for t in tags], axis=0)
            y = Yva[c].to_numpy(); m = ~np.isnan(y)
            pm = .5 * p_tree[c] + .5 * pens
            best, bw = -1, 0
            for w in wg:
                for t in grid:
                    pred = np.where(pm >= t, DEFAULT_VALUES[c], np.power(10., (1-w)*reg_tree[c]+w*ens))
                    a = pw(c, pred[m], y[m]).mean()
                    if a > best: best, bw = a, w
            line.append(f"{name} best w={bw:.2f} acc={best:.4f}")
        print(f"  {c:5s}: " + " | ".join(line))

    print("\n=== 4) 逐井 bootstrap: corrected(tuned) - buggy(tuned) 总分差 ===")
    a1 = tuned_total(BUGGY); a2 = tuned_total(CORR)
    tward = a1[2]; twcor = a2[2]
    def per_well_total(tags, ws, ths):
        ens_all = {c: np.mean([regs[t][c] for t in tags], axis=0) for c in TARGETS}
        pen_all = {c: np.mean([pdfs[t][c] for t in tags], axis=0) for c in TARGETS}
        out = {}
        for lid in pd.unique(larr):
            mw = larr == lid; tot = 0.0
            for c in TARGETS:
                y = Yva[c].to_numpy(); m = (~np.isnan(y)) & mw
                if m.sum() == 0: continue
                pm = .5 * p_tree[c] + .5 * pen_all[c]
                pred = np.where(pm >= ths[c], DEFAULT_VALUES[c], np.power(10., (1-ws[c])*reg_tree[c]+ws[c]*ens_all[c]))
                tot += WEIGHTS[c] * pw(c, pred[m], y[m]).mean()
            out[lid] = tot * 100
        return out
    pb = per_well_total(BUGGY, tward, a1[1]); pc = per_well_total(CORR, twcor, a2[1])
    keys = sorted(pb)
    diff = np.array([pc[k] - pb[k] for k in keys])
    print("  逐井差(井序):", " ".join(f"{d:+.2f}" for d in diff))
    rng = np.random.default_rng(0)
    boot = np.array([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(5000)])
    print(f"  mean diff={diff.mean():+.3f}  boot95%=[{np.percentile(boot,2.5):+.3f},{np.percentile(boot,97.5):+.3f}]  正/负井={int((diff>0).sum())}/{int((diff<0).sum())}")


if __name__ == "__main__":
    main()
