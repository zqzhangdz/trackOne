"""模型自身失配信号能否当分段特征？（sklearn-only，回归器不动，只重训分类器）

核心假设（本项目的新角度）：部署的回归器**只在真实段行上训练**（train.py 两阶段：
分类器选段 → 回归器在真实段上 fit → weight_tol 在真实段上重拟合）。所以在默认段的点上，
回归器的输出是**纯外推**，若外推会系统性地落到训练真实值分布之外，那么「回归输出的离分布
程度」本身就是一个分段信号，而当前管线完全没用它。

先做生死证据（约 1 秒）：回归器在**默认段**上的输出分布 vs 在**真实段**上的输出分布，
二者可分吗？单特征 AUC 若 ≈ 0.5 则这条路线当场就死。

内存纪律：全程 float32；X 转 numpy 后立刻丢弃 DataFrame；每个 fit 完 del + gc.collect()。
clf 用 max_iter=150 做 baseline/candidate 的**对等**头对头比较（部署是 300，绝对值另用
缓存的 300 版 p_def 锚定），避免为了拟合更多而超调迭代数。

用法: python experiments/tunex_mismatch_seg.py
"""
from __future__ import annotations
import gc, os, sys, time, warnings, resource
warnings.filterwarnings("ignore")
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src.feature as F
# v7 特征集（±10/w21 = 225 特征）。src/feature.py 现已被改成 v8（±60/w41 = 321），
# 但 screen_tree_cache 与 clf_retrain_inputs 都是 v7 特征下算的，必须切回 v7 才能对得上。
F.LAG_OFFSETS = [-10, -5, -3, -2, -1, 1, 2, 3, 5, 10]
F.WIN_SIZES = [5, 11, 21]
from src.feature import default_mask, load_wells, make_features, make_targets
from src.model import POR_MONOTONIC, _clf
from src.utils import DEFAULT_VALUES, EPS, TARGETS, WEIGHTS

RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
BND = 657489
N_FEATS_EXPECT = 225
V7 = {"blend_w": {"POR": 0.50, "PERM": 0.50, "SW": 0.75},
      "thresholds": {"POR": 0.130, "PERM": 0.270, "SW": 0.050},
      "clf_tcn_w": {"POR": 0.60, "PERM": 0.70, "SW": 0.30}}
TH_GRID = np.arange(0.01, 0.991, 0.01)
MAX_ITER = 150
T0 = time.time()


def acc_vec(c, pred, true):
    """官方口径逐点准确率向量（PERM 恒 1.0 decade 比值容差、无分母）。"""
    p = np.asarray(pred, np.float64)
    t = np.asarray(true, np.float64)
    if c == "PERM":
        return np.maximum(0.0, 1.0 - np.abs(np.log10(np.maximum(p / np.maximum(t, EPS), EPS))))
    tol = 0.08 if c == "POR" else 0.05
    return np.maximum(0.0, 1.0 - np.abs(p - t) / (tol * (np.abs(t) + EPS)))


def seg_total(acc):
    return float(sum(WEIGHTS[c] * acc[c] for c in TARGETS) * 100.0)


def best_threshold(pm, fd, fr):
    """在 pm 上排序，用前缀和扫阈值，返回 (最优 acc, 最优 th)。"""
    order = np.argsort(pm, kind="stable")
    pos = np.searchsorted(pm[order], TH_GRID, side="left")
    Fd = np.concatenate([[0.0], np.cumsum(fd[order])])
    Fr = np.concatenate([[0.0], np.cumsum(fr[order])])
    n = len(pm)
    acc = (Fr[pos] + (Fd[n] - Fd[pos])) / n
    j = int(np.argmax(acc))
    return float(acc[j]), float(TH_GRID[j])


def pred_at(pm, th, fdv, regb, c):
    """阈值决策后的逐点 acc 向量（默认段用精确默认值）。"""
    out = np.where(pm >= th, DEFAULT_VALUES[c], regb)
    return acc_vec(c, out, fdv)


# ═══════════════════════════════════════════════════════════════════════════
# 载入
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 78)
print("模型失配信号作为分段特征？")
print("=" * 78)
print("载入 80 井 ...", flush=True)
wells = load_wells("./train", has_labels=True)
lens = [len(w) for w in wells]
assert sum(lens) == 730268 and sum(lens[:72]) == BND, (sum(lens), sum(lens[:72]))
df = pd.concat(wells, ignore_index=True)
del wells
gc.collect()
X = make_features(df, context=True, priors=True)
assert X.shape[1] == N_FEATS_EXPECT, f"特征数 {X.shape[1]} != {N_FEATS_EXPECT}（应为 v7 配置）"
Y = make_targets(df)
DM = default_mask(Y)
row_wells = df["logId"].to_numpy()
del df
gc.collect()

X_tr = np.ascontiguousarray(X.iloc[:BND].to_numpy(), dtype=np.float32)
X_va = np.ascontiguousarray(X.iloc[BND:].to_numpy(), dtype=np.float32)
Ytr = {c: Y[c].to_numpy()[:BND] for c in TARGETS}
Yva = {c: Y[c].to_numpy()[BND:] for c in TARGETS}
Dtr = {c: DM[c].to_numpy().astype(bool)[:BND] for c in TARGETS}
Dva = {c: DM[c].to_numpy().astype(bool)[BND:] for c in TARGETS}
nva = len(X_va)
del X, Y, DM
gc.collect()
print(f"  X {X_tr.shape[0]:,}+{nva:,} × {N_FEATS_EXPECT}  [{time.time()-T0:.0f}s]", flush=True)

ck = joblib.load(os.path.join(RUNS, "clf_retrain_inputs.joblib"))
assert ck.get("ver") == 2, "clf_retrain_inputs 必须 ver=2（回归器含 weight_tol）"
rt = {c: np.float32(ck["rt_all"][c]) for c in TARGETS}
p_val_cached = {c: np.float32(ck["p_val"][c]) for c in TARGETS}
del ck
gc.collect()
d = np.load(os.path.join(RUNS, "tcn_all80_parts.npz"))
ens = {c: np.float32(d[f"reg_{c}"]) for c in TARGETS}
pdf = {c: np.float32(d[f"pdf_{c}"]) for c in TARGETS}
del d
gc.collect()

# 验证集混合回归输出（log10 空间）与树回归输出
rt_va = {c: np.float32(rt[c][BND:]) for c in TARGETS}
ens_va = {c: np.float32(ens[c][BND:]) for c in TARGETS}
pdf_va = {c: np.float32(pdf[c][BND:]) for c in TARGETS}
rt_tr = {c: np.float32(rt[c][:BND]) for c in TARGETS}
ens_tr = {c: np.float32(ens[c][:BND]) for c in TARGETS}
regb_va_log = {c: (1 - V7["blend_w"][c]) * rt_va[c] + V7["blend_w"][c] * ens_va[c] for c in TARGETS}
regb_tr_log = {c: (1 - V7["blend_w"][c]) * rt_tr[c] + V7["blend_w"][c] * ens_tr[c] for c in TARGETS}
LG_DEF = {c: float(np.log10(DEFAULT_VALUES[c])) for c in TARGETS}

# ═══════════════════════════════════════════════════════════════════════════
# Phase 0：生死证据
# ═══════════════════════════════════════════════════════════════════════════
QS = [1, 5, 25, 50, 75, 95, 99]
print("\n" + "=" * 78)
print("Phase 0  生死证据：回归输出在「默认段」vs「真实段」的分布（验证 8 井，全 out-of-sample）")
print("=" * 78)
kill_auc = {}
for c in TARGETS:
    v = ~np.isnan(Yva[c])
    m_real = v & ~Dva[c]
    m_def = v & Dva[c]
    r = regb_va_log[c]
    yt = Ytr[c]
    sel_real_tr = ~np.isnan(yt) & ~Dtr[c]
    refA = np.log10(np.maximum(yt[sel_real_tr], EPS))            # 参考 A：训练真实段真值
    refB = regb_tr_log[c][sel_real_tr]                           # 参考 B：训练真实段回归输出
    print(f"\n{c}   默认行 {m_def.sum():,}   真实行 {m_real.sum():,}   有效行 {v.sum():,}")
    for nm, ref in (("A 真值", refA), ("B 回归输出", refB)):
        lo, hi = np.percentile(ref, [1, 99])
        out_lo = np.maximum(lo - r, 0.0)
        out_hi = np.maximum(r - hi, 0.0)
        out = np.maximum(out_lo, out_hi)
        a = float(roc_auc_score(Dva[c][v], out[v]))
        kill_auc[(c, nm)] = a
        print(f"  参考{nm}: log10[p1,p99]=[{lo:+.3f},{hi:+.3f}]  "
              f"超出量 默认段 {out[m_def].mean():.4f} / 真实段 {out[m_real].mean():.4f}   "
              f"单特征 AUC = {a:.4f}")
    print("    默认段 " + " ".join(f"p{q}={np.percentile(r[m_def], q):+.3f}" for q in QS))
    print("    真实段 " + " ".join(f"p{q}={np.percentile(r[m_real], q):+.3f}" for q in QS))
    # 附加：「离默认值多远」能不能区分真实/默认？（真实段真值远离默认，默认段真值就是默认）
    gd = np.abs(r - LG_DEF[c])
    a2 = float(roc_auc_score(Dva[c][v], gd[v]))
    print(f"    |log10(ŷ) − log10(默认)| 的 AUC(默认 vs 真实) = {a2:.4f}  "
          f"(默认段 {gd[m_def].mean():.3f} / 真实段 {gd[m_real].mean():.3f})")

# 0b 参考分位数敏感性：换任意 (lo,hi) 参考区间，单特征 AUC 是否可能抬起来？
#      用来排除「只是 p1/p99 这个参考选得不好」这个可能的辩护。
best_ref = {}
for c in TARGETS:
    yt = Ytr[c]
    sel = ~np.isnan(yt) & ~Dtr[c]
    ref_all = np.log10(np.maximum(yt[sel], EPS))
    rva = regb_va_log[c]
    mv = ~np.isnan(Yva[c])
    ylab = Dva[c][mv]
    bb = (-1.0, 0.0, 0.0, 0)
    for q in range(1, 50):                        # 对称收缩区间 p_q / p_{100-q}
        a0, b0 = np.percentile(ref_all, [q, 100 - q])
        out = np.maximum(np.maximum(a0 - rva, 0.0), np.maximum(rva - b0, 0.0))
        a = float(roc_auc_score(ylab, out[mv]))
        if a > bb[1]:
            bb = (a, float(a0), float(b0), q)
    best_ref[c] = bb
print("\n--- 0b 参考分位数敏感性（2304 个对称区间组合）---")
for c in TARGETS:
    bb = best_ref[c]
    print(f"  {c}: 最优单特征 AUC = {bb[0]:.4f}  参考 log10[{bb[1]:+.3f},{bb[2]:+.3f}] "
          f"(真值 p{bb[3]}/p{100 - bb[3]})")
print(f"  全部网格的最大单特征 AUC = {max(v[0] for v in best_ref.values()):.4f}（0.5 = 随机）")

kill_ok = max(kill_auc.values())
print(f"\n  生死判定：最大单特征 AUC = {kill_ok:.4f}（0.5 = 随机）"
      f"  →  {'可分，继续' if kill_ok > 0.60 else '高度重叠 → 这条路线基本已死，仍跑完拿到硬数字'}")

# ═══════════════════════════════════════════════════════════════════════════
# Phase 1：基线自检（必须落在 73.749 ± 0.05）
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 78)
print("Phase 1  基线自检（v7 配置，纯缓存，不拟合）")
print("=" * 78)
acc_base = {}
comp_base = {}
for c in TARGETS:
    m = ~np.isnan(Yva[c])
    pm = (1 - V7["clf_tcn_w"][c]) * p_val_cached[c] + V7["clf_tcn_w"][c] * pdf_va[c]
    regb = np.power(10.0, regb_va_log[c])
    pred = np.where(pm >= V7["thresholds"][c], DEFAULT_VALUES[c], regb)
    a = float(np.mean(acc_vec(c, pred[m], Yva[c][m])))
    acc_base[c] = a
    comp_base[c] = WEIGHTS[c] * a * 100.0
    print(f"  {c}: acc {a:.4f}  (贡献 {comp_base[c]:.3f})  AUC(p_def) {roc_auc_score(Dva[c][m], pm[m]):.4f}")
tot_base = seg_total(acc_base)
print(f"  >>> 自检 screen total = {tot_base:.3f}   (部署 v7 = 73.749, 允许 73.749 ± 0.05)")
print(f"  >>> {'通过' if abs(tot_base - 73.749) <= 0.05 else '不通过 —— harness 有偏，后续 Δ 不可信'}")

# ═══════════════════════════════════════════════════════════════════════════
# Phase 2：构造失配特征
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 78)
print("Phase 2  构造失配特征（log10 输出空间；参考分布来自训练真实段，推理时可随模型固化）")
print("=" * 78)
WIN = 21


def smooth_dev(r, well_ids, n):
    """|ŷ_log − 邻近窗口均值|，按井分组、按文件内深度顺序（文件内 DEPTH 单调）。"""
    out = np.zeros(n, np.float32)
    _, w = np.unique(np.asarray(well_ids), return_inverse=True)   # 井号 → 井序号
    edges = np.r_[0, np.flatnonzero(np.diff(w)) + 1, n]
    for i in range(len(edges) - 1):
        a, b = int(edges[i]), int(edges[i + 1])
        if a == b:
            continue
        s = pd.Series(r[a:b]).rolling(WIN, center=True, min_periods=1).mean().to_numpy()
        out[a:b] = np.abs(r[a:b] - s).astype(np.float32)
    return out


well_tr = row_wells[:BND]
well_va = row_wells[BND:]
FEAT = {}          # (目标 -> np.ndarray) 训练/验证各一份
REF = {}
for c in TARGETS:
    yt = Ytr[c]
    sel = ~np.isnan(yt) & ~Dtr[c]
    lo, hi = np.percentile(np.log10(np.maximum(yt[sel], EPS)), [1, 99])
    REF[c] = (float(lo), float(hi))
    # 训练侧（回归器对训练行是 in-sample —— 与 §4.11 同折内特征的既有先例一致）
    r_tr = regb_tr_log[c]
    r_va = regb_va_log[c]
    lo, hi = REF[c]
    cols = {}
    cols["ml"] = (np.maximum(lo - r_tr, 0.0).astype(np.float32),
                  np.maximum(lo - r_va, 0.0).astype(np.float32))
    cols["mh"] = (np.maximum(r_tr - hi, 0.0).astype(np.float32),
                  np.maximum(r_va - hi, 0.0).astype(np.float32))
    cols["gd"] = (np.abs(r_tr - LG_DEF[c]).astype(np.float32),
                  np.abs(r_va - LG_DEF[c]).astype(np.float32))
    cols["sm"] = (smooth_dev(r_tr, well_tr, BND), smooth_dev(r_va, well_va, nva))
    print(f"  {c}: log10[p1,p99] = [{lo:+.3f}, {hi:+.3f}]  (默认值 log10 = {LG_DEF[c]:+.3f})")
    FEAT[c] = {k: v for k, v in cols.items()}

# 三目标联合 z 距离（共享特征）
z = {}
for c in TARGETS:
    m = ~np.isnan(Ytr[c]) & ~Dtr[c]
    mu = float(np.mean(regb_tr_log[c][m]))
    sd = float(np.std(regb_tr_log[c][m])) or 1.0
    z[c] = ((regb_tr_log[c] - mu) / sd, (regb_va_log[c] - mu) / sd, mu, sd)
joint_tr = np.sqrt(sum(z[c][0] ** 2 for c in TARGETS)).astype(np.float32)
joint_va = np.sqrt(sum(z[c][1] ** 2 for c in TARGETS)).astype(np.float32)
print(f"  联合 z: 训练真实段 μ/σ = " + ", ".join(f"{c}({z[c][2]:+.3f}/{z[c][3]:.3f})" for c in TARGETS))


def feat_block(c, side):
    """目标 c 的失配特征块；side='tr'/'va'。返回 float32 (n, k)。"""
    i = 0 if side == "tr" else 1
    return np.column_stack([
        FEAT[c]["ml"][i], FEAT[c]["mh"][i],
        FEAT[c]["gd"][i], FEAT[c]["sm"][i],
        joint_tr if side == "tr" else joint_va,
    ])


# ═══════════════════════════════════════════════════════════════════════════
# Phase 3：重训分类器（回归器不动），头对头
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 78)
print(f"Phase 3  重训分类器（max_iter={MAX_ITER}，baseline 与 candidate 对等；回归器/blend_w/clf_tcn_w 全固定）")
print("=" * 78)
VARIANTS = {
    "base   (225)": [],
    "A 自目标距离 (227)": ["ml", "mh"],
    "B +离默认值 (228)": ["ml", "mh", "gd"],
    "C +平滑性 (229)": ["ml", "mh", "sm"],
    "D +联合z (228)": ["ml", "mh", "joint"],
    "ALL (230)": ["ml", "mh", "gd", "sm", "joint"],
}
PMCACHE = os.path.join(RUNS, "mismatch_pm.joblib")
if os.path.exists(PMCACHE):
    _c = joblib.load(PMCACHE)
    if sorted(_c.keys()) == sorted(VARIANTS.keys()):
        pm = {k: {t: np.float32(v[t]) for t in v} for k, v in _c.items()}
        print(f"  复用缓存 {os.path.basename(PMCACHE)}（{len(pm)} 变体 × 3 目标），跳过拟合  [{time.time()-T0:.0f}s]")
        _FITS = False
    else:
        pm = {vn: {} for vn in VARIANTS}
        _FITS = True
else:
    pm = {vn: {} for vn in VARIANTS}
    _FITS = True
if _FITS:
    # 各目标的全长特征块（先算好，再套掩码；训练用 ~isnan，验证全保留、评估时再掩码）
    FB = {vn: {} for vn in VARIANTS}
    for c in TARGETS:
        for vn, keys in VARIANTS.items():
            ks = [k for k in keys if k != "joint"]      # "joint" 是共享特征，不在 FEAT 里
            ft = [FEAT[c][k][0] for k in ks] + ([joint_tr] if "joint" in keys else [])
            fv = [FEAT[c][k][1] for k in ks] + ([joint_va] if "joint" in keys else [])
            k_ = len(ft)
            FB[vn][c] = (np.column_stack(ft).astype(np.float32) if k_ else None,
                         np.column_stack(fv).astype(np.float32) if k_ else None, k_)

    for c in TARGETS:
        vm = ~np.isnan(Ytr[c])
        ylab = Dtr[c][vm].astype(int)
        for vn, keys in VARIANTS.items():
            ft, fv, k_ = FB[vn][c]
            if k_:
                Xt = np.column_stack([X_tr[vm], ft[vm]])
                Xv = np.column_stack([X_va, fv])
            else:
                Xt, Xv = X_tr[vm], X_va
            clf = _clf({"max_iter": MAX_ITER})
            clf.fit(Xt, ylab)
            pv = clf.predict_proba(Xv)[:, 1].astype(np.float32)
            pm[vn][c] = pv
            del clf, Xt, Xv
            gc.collect()
            m = ~np.isnan(Yva[c])
            print(f"  {c:5s} {vn:22s} 特征 {N_FEATS_EXPECT + k_:3d}  "
                  f"AUC {roc_auc_score(Dva[c][m], pv[m]):.4f}"
                  f"  [{time.time()-T0:.0f}s]", flush=True)
        gc.collect()
    joblib.dump(pm, PMCACHE)
    del FB
    gc.collect()

# 基线锚点：缓存里的 300 版 p_def
for c in TARGETS:
    m = ~np.isnan(Yva[c])
    print(f"  [锚点] {c}: 缓存 300 版 p_def AUC = {roc_auc_score(Dva[c][m], p_val_cached[c][m]):.4f}"
          f"  (部署 §4.2 引用 { {'POR':0.8966,'PERM':0.8906,'SW':0.9701}[c]:.4f} )")

# ═══════════════════════════════════════════════════════════════════════════
# Phase 4：官方 acc（v7 阈值）与 λ 融合扫描
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 78)
print("Phase 4  官方 acc @ v7 阈值（回归器/blend_w/clf_tcn_w 全固定，只换分类器）")
print("=" * 78)
acc_v7 = {v: {} for v in VARIANTS}
acc_best = {v: {} for v in VARIANTS}
tot_v7 = {v: 0.0 for v in VARIANTS}
tot_best = {v: 0.0 for v in VARIANTS}
print(f"{'variant':24s}" + "".join(f"{c+' @v7':>16s}" for c in TARGETS)
      + f"{'Total@v7':>11s}{'Total@bestth':>14s}   最优阈值")
best_th_rows = []
for v in VARIANTS:
    totv = 0.0
    totb = 0.0
    bts = []
    for c in TARGETS:
        m = ~np.isnan(Yva[c])
        pmv = (1 - V7["clf_tcn_w"][c]) * pm[v][c] + V7["clf_tcn_w"][c] * pdf_va[c]
        regb = np.power(10.0, regb_va_log[c])
        fdv = Yva[c][m]
        a7 = float(np.mean(acc_vec(c, np.where(pmv[m] >= V7["thresholds"][c],
                                                DEFAULT_VALUES[c], regb[m]), fdv)))
        a_b, th_b = best_threshold(pmv[m],
                                   acc_vec(c, np.full(int(m.sum()), DEFAULT_VALUES[c]), fdv),
                                   acc_vec(c, regb[m], fdv))
        acc_v7[v][c] = a7
        acc_best[v][c] = a_b
        totv += WEIGHTS[c] * a7 * 100.0
        totb += WEIGHTS[c] * a_b * 100.0
        bts.append(f"{c}{th_b:.3f}")
    tot_v7[v] = totv
    tot_best[v] = totb
    best_th_rows.append((v, bts))
    print(f"{v:24s}" + "".join(f"{acc_v7[v][c]:>16.4f}" for c in TARGETS)
          + f"{totv:>11.3f}{totb:>14.3f}   {' '.join(bts)}")

print(f"\n  {'base (225)':24s} 合计 @v7 阈值 = {tot_v7['base   (225)']:.3f}"
      f"   （自检缓存版 {tot_base:.3f}；两者差是 max_iter 150 vs 300，非 pipeline 偏置）")
print(f"  最大候选 Total@v7 = {max(tot_v7.values()):.3f} "
      f"(Δ {max(tot_v7.values()) - tot_v7['base   (225)']:+.3f} vs 本脚本基线)")
print(f"  最大候选 Total@bestth = {max(tot_best.values()):.3f} "
      f"(Δ {max(tot_best.values()) - tot_best['base   (225)']:+.3f})")

# λ 融合扫描：失配分数 vs 现有 p_def（用基线分类器作锚，扫 λ）
print("\n--- λ 融合扫描：p_mix = (1−λ)·p_base + λ·score_out（回归器不动，λ 网格 0.00/0.05/.../1.00）---")
lam = np.arange(0.0, 1.001, 0.05)
lam_res = {}
for c in TARGETS:
    m = ~np.isnan(Yva[c])
    regb = np.power(10.0, regb_va_log[c])
    fdv = Yva[c][m]
    f_def = acc_vec(c, np.full(int(m.sum()), DEFAULT_VALUES[c]), fdv)
    f_reg = acc_vec(c, regb[m], fdv)
    sb = np.maximum(FEAT[c]["ml"][1][m], FEAT[c]["mh"][1][m])
    s = sb / (sb.max() + 1e-9)          # 归一到 [0,1]，与 p_def 同尺度
    ref = (1 - V7["clf_tcn_w"][c]) * p_val_cached[c][m] + V7["clf_tcn_w"][c] * pdf_va[c][m]
    pmv_ref = ref
    row = []
    best = (-1.0, 0.0)
    for l in lam:
        mix = (1 - l) * ref + l * s
        a, th = best_threshold(mix, f_def, f_reg)
        row.append(a)
        if a > best[0]:
            best = (a, float(l))
    lam_res[c] = (row, best, V7["thresholds"][c])
    base_a = float(np.mean(acc_vec(c, np.where(pmv_ref >= V7["thresholds"][c],
                                                DEFAULT_VALUES[c], regb[m]), fdv)))
    print(f"  {c}: λ=0 (现 p_def) acc {row[0]:.4f}  λ*={best[1]:.2f} acc {best[0]:.4f}  "
          f"(v7 阈值 {V7['thresholds'][c]:.3f})")

# ═══════════════════════════════════════════════════════════════════════════
# Phase 5：逐井分解 + 留一井 jackknife
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 78)
print("Phase 5  逐井分解与留一井 jackknife（8 口井才是实验单元）")
print("=" * 78)
va_wells = list(pd.unique(row_wells[BND:]))
mask_well = {w: (row_wells[BND:] == w) for w in va_wells}
best_name = max(tot_v7, key=tot_v7.get)
print(f"  候选 = 最大 Total@v7 的变体：{best_name}")
# 每井「点对总分的贡献」= sum_c WEIGHTS[c] * 100 * (该井有效点上 acc 之和) / (该目标全部有效点数)
# 这样 8 井之和**恒等于**官方逐点等权总分（逐井等权会系统性虚高 +2.31 分，见 §4.8，不可直接引用）
n_valid = {c: int(np.sum(~np.isnan(Yva[c]))) for c in TARGETS}
per_well = {}
for w in va_wells:
    row = {"base": 0.0, "cand": 0.0}
    for c in TARGETS:
        sel = mask_well[w] & ~np.isnan(Yva[c])
        if not sel.any():
            continue
        regb = np.power(10.0, regb_va_log[c])[sel]
        fdv = Yva[c][sel]
        fb = (1 - V7["clf_tcn_w"][c]) * pm["base   (225)"][c][sel] + V7["clf_tcn_w"][c] * pdf_va[c][sel]
        fc = (1 - V7["clf_tcn_w"][c]) * pm[best_name][c][sel] + V7["clf_tcn_w"][c] * pdf_va[c][sel]
        for tag, p in (("base", fb), ("cand", fc)):
            row[tag] += WEIGHTS[c] * 100.0 * float(np.sum(acc_vec(
                c, np.where(p >= V7["thresholds"][c], DEFAULT_VALUES[c], regb), fdv))) / n_valid[c]
    per_well[w] = (row["base"], row["cand"])
print(f"  {'井（点对总分贡献）':34s}{'基线':>9s}{'候选':>9s}{'Δ':>9s}")
for w, (b, k) in per_well.items():
    print(f"  {w:34s}{b:9.3f}{k:9.3f}{k - b:+9.3f}")
deltas = np.array([v[1] - v[0] for v in per_well.values()])
full = float(deltas.sum())
loo = [float(deltas[np.arange(8) != i].sum()) for i in range(8)]
n_pos = int((deltas > 0).sum())
sb = sum(v[0] for v in per_well.values())
sk = sum(v[1] for v in per_well.values())
print(f"  8 井贡献之和：基线 {sb:.3f} / 候选 {sk:.3f}（应与 Phase 4 的 "
      f"{tot_v7['base   (225)']:.3f} / {tot_v7[best_name]:.3f} 一致，逐点等权口径）")
print(f"\n  合计 Δ = {full:+.3f}   正号井 {n_pos}/8   逐井 Δ 范围 [{deltas.min():+.3f}, {deltas.max():+.3f}]")
print(f"  留一井 jackknife [{min(loo):+.3f}, {max(loo):+.3f}]"
      f"  → {'不跨 0' if min(loo) > 0 else '跨 0（方向不可信）'}")
print(f"  screen 噪声 ≈ ±0.3；|Δ| = {abs(full):.3f} → "
      f"{'在噪声带内' if abs(full) < 0.3 else '超出噪声带'}")

_mx = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss   # macOS/BSD: 字节
rss_gb = _mx / 1024.0 ** 3
rss_mb = _mx / 1024.0 ** 2
print(f"\n峰值 RSS {rss_gb:.2f} GB ({rss_mb:,.0f} MB)   "
      f"{'（本轮复用拟合缓存）' if not _FITS else '（含 18 次 HistGB fit）'}   本脚本耗时 {time.time()-T0:.0f}s")
RES = {"tot_base_cache": tot_base, "tot_base_v150": tot_v7["base   (225)"],
       "tot_cand_v150": tot_v7[best_name], "delta_full": full,
       "loo": loo, "n_pos": n_pos, "best_name": best_name,
       "kill_max_auc": kill_ok, "rss_gb": rss_gb,
       "best_ref": {c: v[0] for c, v in best_ref.items()}}
import json
with open(os.path.join(RUNS, "mismatch_seg.json"), "w", encoding="utf-8") as f:
    json.dump(RES, f, ensure_ascii=False, indent=2)
