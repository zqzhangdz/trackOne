"""判定 T12(200ep, 正确 LN10) 是否可信地优于当前部署同源配置(120ep)。

部署同源工件族（正确 LN10）：A=T10+T11(seed42/43,120ep，当前 train.py 的 screen 代理)，
B=T12+T11(seed42 200ep + seed43 120ep)。逐目标在验证井内调优 (w_tcn, threshold, clf_w)，
报告 total/real/def；并用 LOO（7 井调参 -> 留出井）与逐井配对 bootstrap 给出置信度。

阈值加速：acc(t) = mean(f_real·1[pm<t] + f_def·1[pm>=t]) 可按 pm 排序后用前缀和一次算出。
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


def pw(c, pred, true):
    p = np.asarray(pred, float); t = np.asarray(true, float)
    if c == "PERM":
        p = np.log10(np.maximum(p, EPS)); t = np.log10(np.maximum(t, EPS))
    return np.maximum(0.0, 1.0 - np.abs(p - t) / (TOLERANCE[c] * (np.abs(t) + EPS)))


class Scorer:
    def __init__(self):
        wells = load_wells("./train", has_labels=True); ids = [w["logId"].iloc[0] for w in wells]
        va = [w for w in wells if w["logId"].iloc[0] in set(ids[72:])]
        self.va = va
        va_df = pd.concat(va, ignore_index=True)
        self.Y = make_targets(va_df); self.td = default_mask(va_df)
        self.larr = va_df["logId"].to_numpy()
        cache = joblib.load(os.path.join(RUNS, "screen_tree_cache.joblib"))
        self.rt, self.pt = cache["reg"], cache["p_def"]

    def load(self, tags):
        R, P = {}, {}
        for tag in tags:
            a = joblib.load(os.path.join(RUNS, f"tcn_{tag}.joblib")); r, pp = {}, {}
            for w in self.va:
                X = make_features(w, context=False, priors=True).fillna(a["medians"])
                X = ((X - a["stds"].replace(0, 1.0)) / a["stds"].replace(0, 1.0)).to_numpy(np.float32)
                o = numpy_forward(X, a["sd"], a["dilations"])
                for c in TARGETS:
                    r.setdefault(c, []).append(o[c][:, 1]); pp.setdefault(c, []).append(1/(1+np.exp(-o[c][:,0])))
            for c in TARGETS:
                R[c] = R.get(c, 0) + np.concatenate(r[c]) / len(tags)
                P[c] = P.get(c, 0) + np.concatenate(pp[c]) / len(tags)
        return R, P

    def curve(self, c, R, P, mask):
        """返回 (ths, accs, best_cfg) —— 对每个 (cw,w) 用前缀和一次算出所有阈值. """
        y = self.Y[c].to_numpy(); m = (~np.isnan(y)) & mask
        yy = y[m]
        f_def = pw(c, np.full_like(yy, DEFAULT_VALUES[c]), yy)
        rt_m = self.rt[c][m]; pt_m, P_m = self.pt[c][m], P[c][m]
        ths = np.arange(0.02, 0.96, 0.01)
        best = (-1.0, 0.7, 0.5, 0.5)
        for cw in np.arange(0.0, 1.01, 0.1):
            pm = (1 - cw) * pt_m + cw * P_m
            order = np.argsort(pm, kind="stable")
            pm_s = pm[order]
            for w in np.arange(0.0, 1.01, 0.05):
                f_real = pw(c, np.power(10., (1 - w) * rt_m + w * R[c][m]), yy)
                fr_s = f_real[order]
                # acc(t) = (sum_{pm<t} f_real + sum_{pm>=t} f_def)/N
                csum_r = np.concatenate([[0.0], np.cumsum(fr_s)])
                csum_d = np.concatenate([[0.0], np.cumsum(f_def[order])])
                N = len(yy)
                idx = np.searchsorted(pm_s, ths, side="left")  # pm<t 的个数
                acc = (csum_r[idx] + (csum_d[N] - csum_d[idx])) / N
                j = int(np.argmax(acc))
                if acc[j] > best[0]:
                    best = (float(acc[j]), float(w), float(ths[j]), float(cw))
        return best

    def tune(self, R, P, mask):
        return {c: self.curve(c, R, P, mask) for c in TARGETS}

    def score_cfg(self, cfg, R, P, mask):
        tot, parts = 0.0, {}
        for c in TARGETS:
            _, w, t, cw = cfg[c]
            y = self.Y[c].to_numpy(); m = (~np.isnan(y)) & mask
            pm = (1 - cw) * self.pt[c] + cw * P[c]
            pred = np.where(pm >= t, DEFAULT_VALUES[c], np.power(10., (1 - w) * self.rt[c] + w * R[c]))
            acc = pw(c, pred[m], y[m]).mean()
            r = pw(c, pred[m & ~self.td[c].to_numpy()], y[m & ~self.td[c].to_numpy()]).mean()
            d = pw(c, pred[m & self.td[c].to_numpy()], y[m & self.td[c].to_numpy()]).mean()
            parts[c] = (acc, r, d); tot += WEIGHTS[c] * acc
        return tot * 100, parts


def main():
    S = Scorer()
    fam = {"A: T10+T11 (120+120)": ["T10_ln10", "T11_seed43_ln10"],
           "B: T12+T11 (200+120)": ["T12_seed42_200ep", "T11_seed43_ln10"],
           "C: T12+T10 (200+120)": ["T12_seed42_200ep", "T10_ln10"],
           "D: T12+T10+T11": ["T12_seed42_200ep", "T10_ln10", "T11_seed43_ln10"],
           "E: T12+T13 (200+200)": ["T12_seed42_200ep", "T13_seed43_200ep"],
           "F: T12+T13+T11": ["T12_seed42_200ep", "T13_seed43_200ep", "T11_seed43_ln10"],
           "G: all4": ["T10_ln10", "T11_seed43_ln10", "T12_seed42_200ep", "T13_seed43_200ep"]}
    full = np.ones(len(S.larr), bool)
    data = {}
    print("=== in-sample(8 井) tuned ===")
    for name, tags in fam.items():
        R, P = S.load(tags); data[name] = (R, P)
        cfg = S.tune(R, P, full)
        t, parts = S.score_cfg(cfg, R, P, full)
        print(f"  {name:22s} total={t:.3f} real=(" + ",".join(f"{parts[c][1]:.4f}" for c in TARGETS)
              + ") def=(" + ",".join(f"{parts[c][2]:.4f}" for c in TARGETS)
              + ") cfg=" + ",".join(f"{cfg[c][1]:.2f}/{cfg[c][2]:.2f}/{cfg[c][3]:.2f}" for c in TARGETS))

    print("\n=== LOO（7 井调参 -> 留出井评估）===")
    uniq = list(pd.unique(S.larr))
    loo = {}
    for name, (R, P) in data.items():
        vals = []
        for hold in uniq:
            ho = S.larr == hold
            cfg = S.tune(R, P, ~ho)
            t, _ = S.score_cfg(cfg, R, P, ho)
            vals.append(t)
        loo[name] = np.array(vals)
        print(f"  {name:22s} LOO mean={np.nanmean(vals):.3f}  per-well=" + " ".join(f"{v:.1f}" for v in vals))
    A = loo["A: T10+T11 (120+120)"]
    rng = np.random.default_rng(0)
    print("\n=== LOO 相对 A 的配对 bootstrap ===")
    for name, vals in loo.items():
        if name.startswith("A"): continue
        d = vals - A
        boot = np.array([np.nanmean(d[rng.integers(0, len(d), len(d))]) for _ in range(5000)])
        print(f"  {name:24s} d={np.nanmean(d):+.3f} boot95%=[{np.nanpercentile(boot,2.5):+.3f},{np.nanpercentile(boot,97.5):+.3f}] "
              f"pos/neg={int(np.nansum(d>0))}/{int(np.nansum(d<0))}")


if __name__ == "__main__":
    main()
