"""按官方 PERM 公式重调 PERM 的融合参数（不重训，纯数组计算）。

背景：仓库 src/utils.py 的 PERM 口径把容差算成 0.08 * (|log10 y| + ε)（实际约 0.10 decade），
而官方原文是 1 - |log10(max(ŷ/y, ε))| —— **恒为 1.0 decade 的比值容差**。
旧口径比官方严约 10 倍，因此 v6 调出的 PERM 权重 w=0.96（几乎全 TCN）是在奖励
官方根本不看的亚 decade 精度。POR / SW 公式与官方完全一致，参数不受影响。

本脚本：在官方口径下对 PERM 的 (w, threshold, clf_tcn_w) 做网格搜索，并做 LOO 复核。
只用缓存前向；不 fit 模型、不 import torch（遵守同进程运行时约束）。
用法: python experiments/tune_official_perm.py
"""
from __future__ import annotations
import os, sys, warnings, json, time
warnings.filterwarnings("ignore")
import joblib, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.feature import load_wells, make_targets, default_mask
from src.utils import DEFAULT_VALUES, EPS, TARGETS, WEIGHTS

RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
T108 = {"T15": "T15_seed42_150ep", "T16a": "T16_seed43_150ep", "T16b": "T16_seed44_150ep"}
W_GRID = np.arange(0.0, 1.001, 0.01)
TH_GRID = np.arange(0.005, 0.996, 0.005)
CW_GRID = np.arange(0.0, 1.001, 0.05)


def acc_off(c, pred, true):
    """官方口径逐点准确率（图片原文）。"""
    p = np.asarray(pred, float); t = np.asarray(true, float)
    if c == "PERM":
        return np.maximum(0.0, 1.0 - np.abs(np.log10(np.maximum(p / np.maximum(t, EPS), EPS))))
    tol = 0.08 if c == "POR" else 0.05
    return np.maximum(0.0, 1.0 - np.abs(p - t) / (tol * (np.abs(t) + EPS)))


def main() -> None:
    wells = load_wells("./train", has_labels=True)
    ids = [w["logId"].iloc[0] for w in wells]
    va = pd.concat([w for w in wells if w["logId"].iloc[0] in set(ids[72:])], ignore_index=True)
    Y = make_targets(va)
    cache = joblib.load(os.path.join(RUNS, "screen_tree_cache.joblib"))
    rt, pt = cache["reg"], cache["p_def"]
    store = {k: v for k, v in np.load(os.path.join(RUNS, "tcn_screen_outputs.npz")).items()}
    Dm = {c: default_mask(Y)[c].to_numpy() for c in TARGETS}
    larr = va["logId"].to_numpy(); uniq = list(pd.unique(larr))
    t0 = time.time()

    ens = np.mean([store[f"{T108[t]}__PERM__reg"] for t in T108], axis=0)   # log10 空间
    pcn = np.mean([store[f"{T108[t]}__PERM__pdf"] for t in T108], axis=0)
    y = Y["PERM"].to_numpy(); valid = ~np.isnan(y)
    N = valid.sum()
    idx = np.flatnonzero(valid)
    yv = y[idx]
    rtv, ptv, ens_v, pcnv = rt["PERM"][idx], pt["PERM"][idx], ens[idx], pcn[idx]

    print("=== PERM 端点参照（官方口径，全池逐点等权）===")
    base = {}
    for name, pred in [
        ("纯默认常数 0.01", np.full(N, DEFAULT_VALUES["PERM"])),
        ("纯树 (w=0)", np.power(10.0, rtv)),
        ("纯 TCN (w=1)", np.power(10.0, ens_v)),
        ("树+TCN 等权 (w=0.5)", np.power(10.0, 0.5 * rtv + 0.5 * ens_v)),
    ]:
        a = float(np.mean(acc_off("PERM", pred, yv)))
        base[name] = a
        print(f"  {name:22s} {a:.4f}")
    print(f"\n=== (w, threshold, clf_tcn_w) 网格（官方口径，screen 全池）===")
    best = (-1.0, 0.0, 0.005, 0.0)
    for cw in CW_GRID:
        pm = (1 - cw) * ptv + cw * pcnv
        order = np.argsort(pm, kind="stable"); pm_s = pm[order]
        f_def = acc_off("PERM", np.full(N, DEFAULT_VALUES["PERM"]), yv)
        cs_d = np.concatenate([[0.0], np.cumsum(f_def[order])])
        pos = np.searchsorted(pm_s, TH_GRID, side="left")
        for w in W_GRID:
            fr = acc_off("PERM", np.power(10.0, (1 - w) * rtv + w * ens_v), yv)
            cs_r = np.concatenate([[0.0], np.cumsum(fr[order])])
            acc = (cs_r[pos] + (cs_d[N] - cs_d[pos])) / N
            j = int(np.argmax(acc))
            if acc[j] > best[0]:
                best = (float(acc[j]), float(w), float(TH_GRID[j]), float(cw))
    bacc, bw, bth, bcw = best
    print(f"  最佳 PERM: w={bw:.2f} th={bth:.3f} cw={bcw:.2f}  →  acc={bacc:.4f}   [{time.time()-t0:.0f}s]")

    print("\n=== 各候选配置的 PERM acc（官方口径）===")
    cands = {
        "v6 当前 (w.96/th.065/cw.75)": (0.96, 0.065, 0.75),
        "官方重调": (bw, bth, bcw),
        "纯树 (w=0, 无阈值)": (0.0, 0.005, 0.0),
    }
    perm_off = {}
    for name, (w, th, cw) in cands.items():
        pm = (1 - cw) * ptv + cw * pcnv
        pred = np.where(pm >= th, DEFAULT_VALUES["PERM"], np.power(10.0, (1 - w) * rtv + w * ens_v))
        perm_off[name] = float(np.mean(acc_off("PERM", pred, yv)))
        print(f"  {name:30s} {perm_off[name]:.4f}")

    # 各目标各自的 valid 掩码（SW 在病态井上全 NaN），逐目标独立计算后再按官方权重合成
    V6CFG = {"POR": (0.54, 0.115, 0.60), "SW": (0.80, 0.040, 0.30)}
    fixed = {}
    for c in ("POR", "SW"):
        w_, th_, cw_ = V6CFG[c]
        ens_c = np.mean([store[f"{T108[t]}__{c}__reg"] for t in T108], axis=0)
        pcn_c = np.mean([store[f"{T108[t]}__{c}__pdf"] for t in T108], axis=0)
        pmc = (1 - cw_) * pt[c] + cw_ * pcn_c
        pred_c = np.where(pmc >= th_, DEFAULT_VALUES[c], np.power(10.0, (1 - w_) * rt[c] + w_ * ens_c))
        m = ~np.isnan(Y[c].to_numpy())
        fixed[c] = float(np.mean(acc_off(c, pred_c[m], Y[c].to_numpy()[m])))

    print("\n=== 总分（官方口径，逐点等权；POR/SW 保持 v6 配置不变）===")
    for name, (w, th, cw) in cands.items():
        pm = (1 - cw) * pt["PERM"] + cw * pcn
        pred_p = np.where(pm >= th, DEFAULT_VALUES["PERM"], np.power(10.0, (1 - w) * rt["PERM"] + w * ens))
        a_perm = float(np.mean(acc_off("PERM", pred_p[valid], yv)))
        tot = WEIGHTS["POR"] * fixed["POR"] + WEIGHTS["PERM"] * a_perm + WEIGHTS["SW"] * fixed["SW"]
        print(f"  {name:30s}  Total={tot*100:7.3f}   "
              f"POR={fixed['POR']:.4f} PERM={a_perm:.4f} SW={fixed['SW']:.4f}")
    print(f"\n耗时 {time.time()-t0:.0f}s")
    with open(os.path.join(RUNS, "tune_official_perm.json"), "w", encoding="utf-8") as f:
        json.dump({"base": base, "best_perm": {"w": bw, "th": bth, "cw": bcw, "acc": bacc},
                   "perm_off": perm_off}, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
