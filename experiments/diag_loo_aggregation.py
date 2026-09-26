"""核对 LOO 口径：逐井等权 mean vs 官方逐点等权 mean。

官方指标是「全部 N 个深度点独立等权取均值」（src/utils.py::total_score）。
experiments/eval_v5_deployable.py 的 LOO 用 np.nanmean(逐井 total)，即逐井等权，
两者只在「各井点数相等且各井默认/真实段占比相同」时才一致。

本脚本用与 eval_v5_deployable 完全相同的缓存与调参网格，同时输出：
  1) 各留出井的点数、默认/真实段占比、纯默认下界分
  2) LOO 逐井等权 total（原口径）
  3) LOO 逐点等权 total（官方口径，按点数加权汇总三个目标的逐点准确率）
只读缓存，不训练，不 import torch（安全于同进程运行时约束）。
"""
from __future__ import annotations
import os, sys, warnings
warnings.filterwarnings("ignore")
import joblib, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.feature import default_mask, load_wells, make_targets
from src.utils import DEFAULT_VALUES, EPS, TARGETS, TOLERANCE, WEIGHTS

RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
T108 = {"T10": "T10_ln10", "T11": "T11_seed43_ln10", "T14": "T14_seed44_ln10",
        "T15": "T15_seed42_150ep", "T16a": "T16_seed43_150ep", "T16b": "T16_seed44_150ep"}
FAMS = {"A 120x2 (T10,T11)": ["T10", "T11"],
        "V5 150x3 (T15,T16a,T16b)": ["T15", "T16a", "T16b"]}
FAMS_KEYS = {k: [T108[x] for x in v] for k, v in FAMS.items()}


def pw(c, pred, true):
    p = np.asarray(pred, float); t = np.asarray(true, float)
    if c == "PERM":
        p = np.log10(np.maximum(p, EPS)); t = np.log10(np.maximum(t, EPS))
    return np.maximum(0.0, 1.0 - np.abs(p - t) / (TOLERANCE[c] * (np.abs(t) + EPS)))


def main() -> None:
    wells = load_wells("./train", has_labels=True)
    ids = [w["logId"].iloc[0] for w in wells]
    va = [w for w in wells if w["logId"].iloc[0] in set(ids[72:])]
    va_df = pd.concat(va, ignore_index=True); Y = make_targets(va_df); TDEF = default_mask(Y)
    larr = va_df["logId"].to_numpy(); uniq = list(pd.unique(larr))
    cache = joblib.load(os.path.join(RUNS, "screen_tree_cache.joblib"))
    rt, pt = cache["reg"], cache["p_def"]
    store = {k: v for k, v in np.load(os.path.join(RUNS, "tcn_screen_outputs.npz")).items()}
    ths = np.arange(0.02, 0.96, 0.02); wg = np.arange(0.0, 1.01, 0.05); cwg = np.arange(0.0, 1.01, 0.25)

    print("=== 各留出井构成（screen 验证集 8 井）===")
    print(f"{'logId':14s} {'点数':>8s} {'POR_def':>8s} {'PERM_def':>8s} {'SW_def':>8s} {'纯默认分':>9s}")
    floor_pw = {}
    for hold in uniq:
        m = larr == hold
        tot = 0.0; fr = {}
        for c in TARGETS:
            y = Y[c].to_numpy(); mm = m & ~np.isnan(y)
            fr[c] = float(TDEF[c].to_numpy()[mm].mean())
            tot += WEIGHTS[c] * pw(c, np.full_like(y, DEFAULT_VALUES[c])[mm], y[mm]).mean()
        floor_pw[hold] = (int(m.sum()), tot * 100, fr)
        print(f"{hold[:14]:14s} {int(m.sum()):8d} {fr['POR']:8.3f} {fr['PERM']:8.3f} "
              f"{fr['SW']:8.3f} {tot*100:9.3f}")
    agg_floor = (sum(floor_pw[h][0] * floor_pw[h][1] for h in uniq)
                 / sum(floor_pw[h][0] for h in uniq))
    print(f"  纯默认下界  逐点等权={agg_floor:.3f}  逐井等权={np.mean([floor_pw[h][1] for h in uniq]):.3f}")

    def tune(name, tune_mask):
        cfg = {}
        for c in TARGETS:
            tg = FAMS_KEYS[name]
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
                    fr_ = pw(c, np.power(10., (1 - w) * r_m + w * ens[m]), yy)
                    cs_r = np.concatenate([[0.0], np.cumsum(fr_[order])])
                    acc = (cs_r[idx] + (cs_d[N] - cs_d[idx])) / N
                    j = int(np.argmax(acc))
                    if acc[j] > best[0]:
                        best = (float(acc[j]), float(w), float(ths[j]), float(cw))
            cfg[c] = best
        return cfg

    def pred_arrays(name, cfg):
        out = {}
        for c in TARGETS:
            tg = FAMS_KEYS[name]
            ens = np.mean([store[f"{t}__{c}__reg"] for t in tg], axis=0)
            pcn = np.mean([store[f"{t}__{c}__pdf"] for t in tg], axis=0)
            _, w, t, cw = cfg[c]
            pm = (1 - cw) * pt[c] + cw * pcn
            pred = np.where(pm >= t, DEFAULT_VALUES[c], np.power(10., (1 - w) * rt[c] + w * ens))
            y = Y[c].to_numpy(); f = pw(c, pred, np.where(np.isnan(y), pred, y))
            out[c] = (f, ~np.isnan(y))
        return out

    print("\n=== LOO：两种汇总口径对比（同一批预测，仅聚合方式不同）===")
    for name in FAMS:
        per_well = []; num = 0.0; den = 0
        for hold in uniq:
            m = larr == hold
            cfg = tune(name, ~m)
            p = pred_arrays(name, cfg)
            per_well.append(sum(WEIGHTS[c] * p[c][0][m].mean() for c in TARGETS) * 100)
            for c in TARGETS:
                mm = m & p[c][1]
                num += WEIGHTS[c] * p[c][0][mm].sum(); den += WEIGHTS[c] * mm.sum()
        official = num / den * 100
        print(f"  {name:26s} 逐井等权={np.mean(per_well):7.3f}  官方逐点等权={official:7.3f}  "
              f"差={np.mean(per_well)-official:+.3f}")
        print(f"      逐井分=" + " ".join(f"{x:5.1f}" for x in per_well))


if __name__ == "__main__":
    main()
