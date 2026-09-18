"""v5 判定：可部署族（末轮≈最优，无需 epoch 选择）的 screen + LOO 综合评估。

可部署成员（实测 best 与 last 差 <0.01 或 last 即 best）：
  T10(120,se42) T11(120,se43) T14(120,se44) T15(150,se42) T16a(150,se43) T16b(150,se44)
对比 T12(200,se42)：best@132=69.058 vs last=68.927，差 0.131 → 需 epoch 选择才可部署。

报告 (a) 固定 v4 权重 screen；(b) 验证集内逐目标调优 screen（含 cfg）；(c) LOO + bootstrap。
LOO 严格为「其余 7 井调参 -> 留出井评估」，不发生调参泄漏。
"""
from __future__ import annotations
import os, sys, warnings
warnings.filterwarnings("ignore")
import joblib, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.feature import default_mask, load_wells, make_targets
from src.utils import DEFAULT_VALUES, EPS, TARGETS, TOLERANCE, WEIGHTS

RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
V4W = {"POR": 0.60, "PERM": 0.95, "SW": 0.75}
V4T = {"POR": 0.10, "PERM": 0.05, "SW": 0.05}
CLFW = 0.5
T108 = {"T10": "T10_ln10", "T11": "T11_seed43_ln10", "T14": "T14_seed44_ln10",
        "T15": "T15_seed42_150ep", "T16a": "T16_seed43_150ep", "T16b": "T16_seed44_150ep",
        "T12": "T12_seed42_200ep"}


def pw(c, pred, true):
    p = np.asarray(pred, float); t = np.asarray(true, float)
    if c == "PERM":
        p = np.log10(np.maximum(p, EPS)); t = np.log10(np.maximum(t, EPS))
    return np.maximum(0.0, 1.0 - np.abs(p - t) / (TOLERANCE[c] * (np.abs(t) + EPS)))


def main():
    wells = load_wells("./train", has_labels=True); ids = [w["logId"].iloc[0] for w in wells]
    va = [w for w in wells if w["logId"].iloc[0] in set(ids[72:])]
    va_df = pd.concat(va, ignore_index=True); Y = make_targets(va_df); TDEF = default_mask(Y)
    larr = va_df["logId"].to_numpy(); uniq = list(pd.unique(larr))
    cache = joblib.load(os.path.join(RUNS, "screen_tree_cache.joblib")); rt, pt = cache["reg"], cache["p_def"]
    store = {k: v for k, v in np.load(os.path.join(RUNS, "tcn_screen_outputs.npz")).items()}
    full = np.ones(len(larr), bool)
    ths = np.arange(0.02, 0.96, 0.02); wg = np.arange(0.0, 1.01, 0.05); cwg = np.arange(0.0, 1.01, 0.25)

    fams = {
        "A  120x2 (T10,T11)": ["T10", "T11"],
        "120x3 (T10,T11,T14)": ["T10", "T11", "T14"],
        "150x3 (T15,T16a,T16b)": ["T15", "T16a", "T16b"],
        "150x3+T14": ["T15", "T16a", "T16b", "T14"],
        "150x3+T10+T11": ["T15", "T16a", "T16b", "T10", "T11"],
        "all6 (120x3+150x3)": ["T10", "T11", "T14", "T15", "T16a", "T16b"],
        "T12+T10+T11": ["T12", "T10", "T11"],
    }
    tags = {k: [T108[x] for x in v] for k, v in fams.items()}

    def tune(name, tune_mask):
        """在 tune_mask 上逐目标调 (w, th, clf_w)；返回 cfg。"""
        cfg = {}
        for c in TARGETS:
            tg = tags[name]
            ens = np.mean([store[f"{t}__{c}__reg"] for t in tg], axis=0)
            pcn = np.mean([store[f"{t}__{c}__pdf"] for t in tg], axis=0)
            y = Y[c].to_numpy(); m = (~np.isnan(y)) & tune_mask
            yy = y[m]; N = len(yy)
            f_def = pw(c, np.full_like(yy, DEFAULT_VALUES[c]), yy)
            r_m, p_m, pn_m = rt[c][m], pt[c][m], pcn[m]
            best = (-1.0, 0.70, 0.10, 0.50)
            for cw in cwg:
                pm = (1 - cw) * p_m + cw * pn_m
                order = np.argsort(pm, kind="stable"); pm_s = pm[order]
                cs_d = np.concatenate([[0.0], np.cumsum(f_def[order])])
                idx = np.searchsorted(pm_s, ths, side="left")
                for w in wg:
                    fr = pw(c, np.power(10., (1 - w) * r_m + w * ens[m]), yy)
                    cs_r = np.concatenate([[0.0], np.cumsum(fr[order])])
                    acc = (cs_r[idx] + (cs_d[N] - cs_d[idx])) / N
                    j = int(np.argmax(acc))
                    if acc[j] > best[0]:
                        best = (float(acc[j]), float(w), float(ths[j]), float(cw))
            cfg[c] = best
        return cfg

    def score(name, cfg, eval_mask):
        """用 cfg 在 eval_mask 上评估，返回 (total, parts)。"""
        tot = 0.0; parts = {}
        for c in TARGETS:
            tg = tags[name]
            ens = np.mean([store[f"{t}__{c}__reg"] for t in tg], axis=0)
            pcn = np.mean([store[f"{t}__{c}__pdf"] for t in tg], axis=0)
            y = Y[c].to_numpy(); m = (~np.isnan(y)) & eval_mask
            _, w, t, cw = cfg[c]
            pm = (1 - cw) * pt[c] + cw * pcn
            pred = np.where(pm >= t, DEFAULT_VALUES[c], np.power(10., (1 - w) * rt[c] + w * ens))
            f = pw(c, pred[m], y[m]); td = TDEF[c].to_numpy()[m]
            parts[c] = (f.mean(), f[~td].mean() if (~td).sum() else np.nan, f[td].mean() if td.sum() else np.nan)
            tot += WEIGHTS[c] * f.mean()
        return tot * 100, parts

    def score_fixed(name, eval_mask):
        tot = 0.0; parts = {}
        for c in TARGETS:
            tg = tags[name]
            ens = np.mean([store[f"{t}__{c}__reg"] for t in tg], axis=0)
            pcn = np.mean([store[f"{t}__{c}__pdf"] for t in tg], axis=0)
            y = Y[c].to_numpy(); m = (~np.isnan(y)) & eval_mask
            pm = (1 - CLFW) * pt[c] + CLFW * pcn
            pred = np.where(pm >= V4T[c], DEFAULT_VALUES[c], np.power(10., (1 - V4W[c]) * rt[c] + V4W[c] * ens))
            f = pw(c, pred[m], y[m]); td = TDEF[c].to_numpy()[m]
            parts[c] = (f.mean(), f[~td].mean() if (~td).sum() else np.nan, f[td].mean() if td.sum() else np.nan)
            tot += WEIGHTS[c] * f.mean()
        return tot * 100, parts

    print("=== (a) 固定 v4 权重 screen（与 68.576 基线同口径）===")
    A_fixed = None
    for name in tags:
        v, parts = score_fixed(name, full)
        if A_fixed is None: A_fixed = v
        print(f"  {name:22s} {v:7.3f} ({v-A_fixed:+.3f})  real=({parts['POR'][1]:.4f},{parts['PERM'][1]:.4f},{parts['SW'][1]:.4f})")

    print("\n=== (b) 验证集内逐目标调优 screen（in-sample，仅参考）===")
    A_tuned = None
    for name in tags:
        cfg = tune(name, full); t, parts = score(name, cfg, full)
        if A_tuned is None: A_tuned = t
        print(f"  {name:22s} {t:7.3f} ({t-A_tuned:+.3f})  real=({parts['POR'][1]:.4f},{parts['PERM'][1]:.4f},{parts['SW'][1]:.4f})"
              f"  cfg=" + ",".join(f"{cfg[c][1]:.2f}/{cfg[c][2]:.2f}/{cfg[c][3]:.2f}" for c in TARGETS))

    print("\n=== (c) LOO（7井调参 -> 留出井评估；逐井等权 total）===")
    loo = {}
    for name in tags:
        vals = []
        for hold in uniq:
            m = larr == hold
            cfg = tune(name, ~m)
            v, _ = score(name, cfg, m)
            vals.append(v)
        loo[name] = np.array(vals)
        print(f"  {name:22s} mean={np.nanmean(vals):.3f}  per-well=" + " ".join(f"{x:.1f}" for x in vals))
    A = loo["A  120x2 (T10,T11)"]; rng = np.random.default_rng(0)
    print("\n=== LOO 相对 A（配对 bootstrap 8000x）===")
    for name, v in loo.items():
        if name.startswith("A "): continue
        d = v - A
        boot = np.array([np.nanmean(d[rng.integers(0, len(d), len(d))]) for _ in range(8000)])
        print(f"  {name:22s} d={np.nanmean(d):+.3f} boot95=[{np.nanpercentile(boot,2.5):+.3f},{np.nanpercentile(boot,97.5):+.3f}] "
              f"pos/neg={int(np.nansum(d>0))}/{int(np.nansum(d<0))}")


if __name__ == "__main__":
    main()
