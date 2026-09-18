"""留一井(LOO)稳健性评估：融合权重/阈值在 7 口验证井上调优、在留出的 1 口井上评估.

动机：v4 的逐目标融合权重与决策阈值与评分共用同一 8 井验证集，存在选择偏差（
PROJECT_SUMMARY §6.1-4 已列为缺陷）。本脚本给出"调优协议本身"的无偏估计。
纯 numpy + joblib，不导入 torch（避免 sklearn/torch OpenMP 冲突）。

高效实现：固定 (w,cw) 后把 p_def 阈值扫描写成排序 + 前缀和，得到 (阈值 × 井) 准确率矩阵。
"""
from __future__ import annotations
import os, sys, warnings, argparse
warnings.filterwarnings("ignore")
import joblib, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.feature import default_mask, load_wells, make_features, make_targets
from src.utils import DEFAULT_VALUES, EPS, TARGETS, TOLERANCE, WEIGHTS
from solution.tcn_model import numpy_forward

RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
TH_GRID = np.arange(0.02, 0.71, 0.02)

def pointwise(c, pred, true):
    p = np.asarray(pred, float); t = np.asarray(true, float)
    if c == "PERM":
        p = np.log10(np.maximum(p, EPS)); t = np.log10(np.maximum(t, EPS))
    return np.maximum(0.0, 1.0 - np.abs(p - t) / (TOLERANCE[c] * (np.abs(t) + EPS)))

def build(tags):
    wells = load_wells("./train", has_labels=True)
    ids = [w["logId"].iloc[0] for w in wells]
    va = [w for w in wells if w["logId"].iloc[0] in set(ids[72:])]
    va_df = pd.concat(va, ignore_index=True)
    Yva = make_targets(va_df)
    cache = joblib.load(os.path.join(RUNS, "screen_tree_cache.joblib"))
    reg_tree, p_tree = cache["reg"], cache["p_def"]
    reg_by, p_by = {}, {}
    for tag in tags:
        art = joblib.load(os.path.join(RUNS, f"tcn_{tag}.joblib"))
        r, pp = {}, {}
        for w in va:
            Xw = make_features(w, context=False, priors=True).fillna(art["medians"])
            Xw = ((Xw - art["stds"].replace(0, 1.0)) / art["stds"].replace(0, 1.0)).to_numpy(np.float32)
            o = numpy_forward(Xw, art["sd"], art["dilations"])
            for c in TARGETS:
                r.setdefault(c, []).append(o[c][:, 1]); pp.setdefault(c, []).append(1/(1+np.exp(-o[c][:,0])))
        reg_by[tag] = {c: np.concatenate(v) for c, v in r.items()}
        p_by[tag] = {c: np.concatenate(v) for c, v in pp.items()}
    print(f"loaded {len(tags)} TCN seeds: {tags}")
    ens = {c: np.mean([reg_by[t][c] for t in tags], axis=0) for c in TARGETS}
    p_ens = {c: np.mean([p_by[t][c] for t in tags], axis=0) for c in TARGETS}
    return dict(Yva=Yva, wit=va_df["logId"].to_numpy(), well_ids=[w["logId"].iloc[0] for w in va],
                reg_tree=reg_tree, p_tree=p_tree, ens=ens, p_ens=p_ens)

def acc_matrix(D, c, w, cw):
    """返回 (n_th, n_well) 准确率矩阵；行=阈值，列=井。"""
    y = D["Yva"][c].to_numpy(); m = ~np.isnan(y)
    reg = (1-w)*D["reg_tree"][c] + w*D["ens"][c]
    pm = (1-cw)*D["p_tree"][c] + cw*D["p_ens"][c]
    dacc = pointwise(c, np.full(int(m.sum()), DEFAULT_VALUES[c]), y[m])
    racc = pointwise(c, np.power(10.0, reg[m]), y[m])
    p = pm[m]; wt = D["wit"][m]
    order = np.argsort(p); dacc, racc, p, wt = dacc[order], racc[order], p[order], wt[order]
    wu = list(pd.unique(D["well_ids"])); widx = {lid:i for i,lid in enumerate(wu)}
    wi = np.array([widx[x] for x in wt])
    n = len(p)
    cnt = np.zeros((len(wu), n+1)); sd = np.zeros((len(wu), n+1)); sr = np.zeros((len(wu), n+1))
    for i in range(len(wu)):
        sel = (wi == i).astype(float)
        cnt[i,1:] = np.cumsum(sel); sd[i,1:] = np.cumsum(sel*dacc); sr[i,1:] = np.cumsum(sel*racc)
    out = np.zeros((len(TH_GRID), len(wu)))
    for j, t in enumerate(TH_GRID):
        k = int(np.searchsorted(p, t, side="left"))
        s_def = sd[:, n] - sd[:, k]; s_reg = sr[:, n] - sr[:, k]
        out[j] = (s_def + s_reg) / np.maximum(cnt[:, n], 1)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default="T6_long,T8_seed43,T9_seed44")
    ap.add_argument("--w_grid", default="0.5,0.7,0.9,0.95,1.0")
    ap.add_argument("--clf_grid", default="0.0,0.3,0.5,0.7")
    a = ap.parse_args()
    tags = [t.strip() for t in a.tags.split(",") if t.strip()]
    D = build(tags)
    w_grid = [float(x) for x in a.w_grid.split(",")]
    cw_grid = [float(x) for x in a.clf_grid.split(",")]
    wells = D["well_ids"]
    print(f"wells({len(wells)}): {[w[:8] for w in wells]}")

    # 预计算：每个目标 × 每个 (w,cw) -> (n_th, n_well)
    tab = {c: {} for c in TARGETS}
    for c in TARGETS:
        for w in w_grid:
            for cw in cw_grid:
                tab[c][(w,cw)] = acc_matrix(D, c, w, cw)
    print("precomputed acc matrices")

    def best_on(c, subset_idx):
        best = (-1, None)
        for key, mat in tab[c].items():
            v = mat[:, subset_idx].mean(axis=1)
            j = int(np.argmax(v))
            if v[j] > best[0]: best = (float(v[j]), (key[0], key[1], float(TH_GRID[j])))
        return best[1]

    def score_on(c, cfg, subset_idx):
        w, cw, t = cfg; j = int(np.argmin(np.abs(TH_GRID - t)))
        return float(tab[c][(w,cw)][j, subset_idx].mean())

    all_idx = np.arange(len(wells))
    print("\n--- 全 8 井调优（现状口径） ---")
    full = {c: best_on(c, all_idx) for c in TARGETS}
    for c in TARGETS:
        print(f"  {c}: w_tcn={full[c][0]:.2f} clf_w={full[c][1]:.2f} th={full[c][2]:.2f}")
    full_total = sum(WEIGHTS[c]*score_on(c, full[c], all_idx) for c in TARGETS)*100
    print(f"  full-8 tuned total = {full_total:.3f}")

    print("\n--- LOO（7 井调优 -> 留出 1 井）---")
    loo = []
    for hi, h in enumerate(wells):
        rest = np.array([i for i in range(len(wells)) if i != hi])
        tot = 0.0; cfg = {}
        for c in TARGETS:
            cf = best_on(c, rest); cfg[c] = cf
            tot += WEIGHTS[c]*score_on(c, cf, np.array([hi]))
        loo.append(tot*100)
        print(f"  hold {h[:8]}: total={tot*100:.3f} w={ {k:cfg[k][0] for k in TARGETS} } cw={ {k:cfg[k][1] for k in TARGETS} }")
    loo = np.array(loo)
    print(f"\nLOO tuned-config: {loo.mean():.3f} ± {loo.std():.3f}  |  in-sample(同 8 井) = {full_total:.3f}")
    print(f"调优乐观偏差 = {full_total - loo.mean():+.3f}")

if __name__ == "__main__":
    main()
