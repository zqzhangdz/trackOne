"""v6 前置：不重训，对已训练 TCN 族做逐目标 (w, threshold, clf_w) 精调。

用途：
  1) 确定 v6 训练时应携带的融合配置（w/th/cw 与 TCN 权重无关，可先定后训）
  2) 给出「当前可部署成员的现实可达总分」上限估计

口径：screen 8 井、逐目标独立调优、cumsum+searchsorted（同 eval_v5_deployable.py）。
指标：src/utils 官方实现；总分用 screen 全池逐点等权（= 官方聚合），不做逐井等权。
LOO 复核：固定已选配置（不在留出井上重调），检查相对 v5 配置是否同向，防 screen 过拟合。
只用缓存前向，不训练，不 import torch。
用法: python experiments/tune_v6_grid.py
"""
from __future__ import annotations
import os, sys, warnings, json, time
warnings.filterwarnings("ignore")
import joblib, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.feature import default_mask, load_wells, make_targets
from src.utils import DEFAULT_VALUES, EPS, TARGETS, TOLERANCE, WEIGHTS

RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
T108 = {"T10": "T10_ln10", "T11": "T11_seed43_ln10", "T14": "T14_seed44_ln10",
        "T15": "T15_seed42_150ep", "T16a": "T16_seed43_150ep", "T16b": "T16_seed44_150ep"}
FAMS = {
    "A 120x2 (T10,T11)":        ["T10", "T11"],
    "V5 150x3 (T15,T16a,T16b)": ["T15", "T16a", "T16b"],
    "150x3 + T14":              ["T15", "T16a", "T16b", "T14"],
    "all6 (120x3 + 150x3)":     ["T10", "T11", "T14", "T15", "T16a", "T16b"],
}
KEYS = {k: [T108[x] for x in v] for k, v in FAMS.items()}
V5CFG = {"POR": (0.55, 0.12, 0.55), "PERM": (0.95, 0.07, 0.55), "SW": (0.85, 0.04, 0.55)}

W_GRID = np.arange(0.0, 1.001, 0.02)
TH_GRID = np.arange(0.005, 0.996, 0.005)
CW_GRID = np.arange(0.0, 1.001, 0.05)


def pw(c, pred, true):
    p = np.asarray(pred, float); t = np.asarray(true, float)
    if c == "PERM":
        p = np.log10(np.maximum(p, EPS)); t = np.log10(np.maximum(t, EPS))
    return np.maximum(0.0, 1.0 - np.abs(p - t) / (TOLERANCE[c] * (np.abs(t) + EPS)))


def build(name, Y, rt, pt, store, Dm, w, th, cw):
    """给定配置，返回 {目标: (逐点acc数组, valid掩码)}。"""
    tg = KEYS[name]
    out = {}
    for c in TARGETS:
        ens = np.mean([store[f"{t}__{c}__reg"] for t in tg], axis=0)
        pcn = np.mean([store[f"{t}__{c}__pdf"] for t in tg], axis=0)
        y = Y[c].to_numpy()
        pm = (1 - cw) * pt[c] + cw * pcn
        pred = np.where(pm >= th, DEFAULT_VALUES[c], np.power(10.0, (1 - w) * rt[c] + w * ens))
        out[c] = (pw(c, pred, np.where(np.isnan(y), pred, y)), ~np.isnan(y))
    return out


def main() -> None:
    wells = load_wells("./train", has_labels=True)
    ids = [w["logId"].iloc[0] for w in wells]
    va = [w for w in wells if w["logId"].iloc[0] in set(ids[72:])]
    va_df = pd.concat(va, ignore_index=True)
    Y = make_targets(va_df)
    cache = joblib.load(os.path.join(RUNS, "screen_tree_cache.joblib"))
    rt, pt = cache["reg"], cache["p_def"]
    store = {k: v for k, v in np.load(os.path.join(RUNS, "tcn_screen_outputs.npz")).items()}
    Dm = {c: default_mask(Y)[c].to_numpy() for c in TARGETS}
    larr = va_df["logId"].to_numpy(); uniq = list(pd.unique(larr))
    full = np.ones(len(va_df), bool)
    t0 = time.time()

    def total_of(name, cfg, mask=full):
        tot = 0.0; accs = {}
        for c in TARGETS:
            w, th, cw = cfg[c]
            f, valid = build(name, Y, rt, pt, store, Dm, w, th, cw)[c]
            m = mask & valid
            accs[c] = f[m].mean(); tot += WEIGHTS[c] * accs[c]
        return tot * 100, accs

    print("=== screen 纯默认下界（逐点等权 = 官方聚合）===")
    floor = 0.0
    for c in TARGETS:
        y = Y[c].to_numpy(); m = full & ~np.isnan(y)
        a = pw(c, np.full_like(y, DEFAULT_VALUES[c])[m], y[m]).mean()
        floor += WEIGHTS[c] * a
        print(f"  {c:5s} acc={a:.4f}")
    print(f"  纯默认下界 = {floor*100:.3f}")

    def tune_one(c, tg):
        ens = np.mean([store[f"{t}__{c}__reg"] for t in tg], axis=0)
        pcn = np.mean([store[f"{t}__{c}__pdf"] for t in tg], axis=0)
        y = Y[c].to_numpy(); m = full & ~np.isnan(y)
        yy = y[m]; N = len(yy)
        f_def = pw(c, np.full_like(yy, DEFAULT_VALUES[c]), yy)
        r_m, p_m, pn_m = rt[c][m], pt[c][m], pcn[m]
        best = (-1.0, 0.5, 0.10, 0.5)
        for cw in CW_GRID:
            pm = (1 - cw) * p_m + cw * pn_m
            order = np.argsort(pm, kind="stable"); pm_s = pm[order]
            cs_d = np.concatenate([[0.0], np.cumsum(f_def[order])])
            idx = np.searchsorted(pm_s, TH_GRID, side="left")
            for w in W_GRID:
                fr = pw(c, np.power(10.0, (1 - w) * r_m + w * ens[m]), yy)
                cs_r = np.concatenate([[0.0], np.cumsum(fr[order])])
                acc = (cs_r[idx] + (cs_d[N] - cs_d[idx])) / N
                j = int(np.argmax(acc))
                if acc[j] > best[0]:
                    best = (float(acc[j]), float(w), float(TH_GRID[j]), float(cw))
        return {"w": best[1], "th": best[2], "cw": best[3], "acc": best[0]}

    print("\n=== 逐目标精调（screen 全池，官方逐点等权）===")
    results = {"floor_screen": floor * 100}
    for name in FAMS:
        cfg = {}
        for c in TARGETS:
            t = tune_one(c, KEYS[name])
            cfg[c] = (t["w"], t["th"], t["cw"])
        tot, accs = total_of(name, cfg)
        real = {}
        for c in TARGETS:
            w, th, cw = cfg[c]
            f, valid = build(name, Y, rt, pt, store, Dm, w, th, cw)[c]
            mm = full & valid & (~Dm[c])
            real[c] = f[mm].mean()
        v5tot, _ = total_of(name, V5CFG)
        line = (f"  {name:28s} 精调={tot:7.3f}  v5配置={v5tot:7.3f}  Δ={tot-v5tot:+.3f}  "
                f"real=(" + ",".join(f"{real[c]:.4f}" for c in TARGETS) + ")  cfg=" +
                ",".join(f"{c} w{cfg[c][0]:.2f}/th{cfg[c][1]:.3f}/cw{cfg[c][2]:.2f}" for c in TARGETS))
        print(line + f"   [{time.time()-t0:.0f}s]")
        results[name] = {"tuned_total": tot, "v5cfg_total": v5tot,
                         "cfg": {c: {"w": cfg[c][0], "th": cfg[c][1], "cw": cfg[c][2]} for c in TARGETS},
                         "real": real}

    print("\n=== LOO 复核：固定精调配置（不在留出井重调）vs v5 配置 ===")
    print(f"{'族':28s} {'v5(逐井)':>10s} {'精调(逐井)':>10s} {'v5(逐点)':>10s} {'精调(逐点)':>10s} {'Δ逐点':>8s}")
    for name in ["A 120x2 (T10,T11)", "V5 150x3 (T15,T16a,T16b)", "150x3 + T14"]:
        row = {}
        for label, cfgmap in [("v5", V5CFG), ("tuned", {c: cfg[c] for c in TARGETS})]:
            per_well = []; num = 0.0; den = 0
            for hold in uniq:
                m = larr == hold
                tot = 0.0
                for c in TARGETS:
                    w, th, cw = cfgmap[c]
                    f, valid = build(name, Y, rt, pt, store, Dm, w, th, cw)[c]
                    mm = m & valid
                    tot += WEIGHTS[c] * f[mm].mean()
                    num += WEIGHTS[c] * f[mm].sum(); den += WEIGHTS[c] * mm.sum()
                per_well.append(tot * 100)
            row[label] = (float(np.mean(per_well)), num / den * 100)
        print(f"  {name:26s} {row['v5'][0]:10.3f} {row['tuned'][0]:10.3f} "
              f"{row['v5'][1]:10.3f} {row['tuned'][1]:10.3f} {row['tuned'][1]-row['v5'][1]:+8.3f}   "
              f"[{time.time()-t0:.0f}s]")

    with open(os.path.join(RUNS, "tune_v6_grid.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n耗时 {time.time()-t0:.0f}s  配置已写 experiments/runs/tune_v6_grid.json")


if __name__ == "__main__":
    main()
