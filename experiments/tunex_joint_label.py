"""联合默认标签（all-three-default / all-non-default / 4 分类）是否比单目标标签更可分。

动机：默认占比 POR 0.6595 / PERM 0.6691 / SW 0.7009，而"三目标同时默认"占 65.92%，
混合态只有 4.4%（0 个默认 29.6% / 1 个 3.7% / 2 个 0.7% / 3 个 65.9%）。
现行方案逐目标各训一个分类器（AUC 0.8966 / 0.8906 / 0.9701，72 训 / 8 验）。
§4.11 试过"跨目标 p_def 作特征"，POR/PERM 的 AUC 反降（0.8966→0.8915/0.8918）。

本轮假设：单目标分段信号互相不共享，但**联合标签**（all-three-default）是一个更
"干净"的几何边界 —— 可能被同一组 225 个特征分得更开。若真能分得更开，就用
p_joint 与单目标 p_def 加权融合 (1-λ)p_def + λ·p_joint，在验证集扫 λ 看三目标官方 acc。

判定标准（比 §6 更严）：**AUC 必须真升才算数**。阈值随 λ 重扫属于二次选择，
会系统性抬高 acc，只作辅助读数；AUC 阈值无关，是主证据。

评估口径与部署完全一致：树回归器带 weight_tol 重训（train.py:278-290），
按 v7 的 blend_w / clf_tcn_w 与缓存的 TCN 前向融合，阈值在验证集 0.01..0.99 扫。
特征固定 ±10/w21（225 列，v7 部署配置），**不是** v8 的 ±60/w41。

用法:
  python experiments/tunex_joint_label.py             # 主实验（约 12 min）
  python experiments/tunex_joint_label.py --jack      # + 留一井 jackknife（再加约 25 min）
"""
from __future__ import annotations
import gc
import os
import resource
import sys
import time
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src.feature as F
from src.feature import default_mask, load_wells, make_targets, transform_target
from src.model import POR_MONOTONIC, _clf, _reg
from src.utils import DEFAULT_VALUES, EPS, TARGETS, WEIGHTS

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")
FEAT_CACHE = os.path.join(RUNS, "joint_label_feat225.npz")
N_TRAIN_WELLS = 72
LAGS = [-10, -5, -3, -2, -1, 1, 2, 3, 5, 10]
WINS = [5, 11, 21]
# 分类器迭代数：任务上限 150（部署是 300）。代价是单目标基线 AUC 略低于部署
# （POR 0.8955 vs 0.8966 实测），但所有候选同参数，AUCΔ 是可比口径。
CLF_ITERS = 100
V7 = {"blend_w": {"POR": 0.50, "PERM": 0.50, "SW": 0.75},
      "thresholds": {"POR": 0.13, "PERM": 0.27, "SW": 0.05},
      "clf_tcn_w": {"POR": 0.60, "PERM": 0.70, "SW": 0.30}}
TH_GRID = np.arange(0.01, 0.991, 0.01)
LAM_GRID = np.round(np.arange(0.0, 1.0001, 0.05), 2)
T0 = time.time()
_XVA = None          # fit_clf 的验证输入（模块级，省一次参数穿透）


def rss_gb() -> float:
    """返回峰值 RSS（GB，高水位，非当前值）。

    ⚠️ 单位陷阱（本轮踩过）：POSIX 说 ru_maxrss 是 KB，但 **macOS 实测是字节**。
    判据：本环境裸 python+numpy 的 ru_maxrss 是 26,411,008 —— 按 KB 读是 26 GB，
    超过 16 GB 物理内存，不可能；按字节读是 25 MB，符合。所以 macOS 上是 /1e9。
    旧写法 /1e6 把 GB 报成 MB 的 1000 倍，实测 4 分类那次的 "1904.85 GB" 实为 1.90 GB。
    这里按量级自判单位，两种系统都能用。
    """
    v = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return v / 1e9 if v > 100_000_000 else v / 1024 / 1024 / 1024


def acc_vec(c: str, pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    p = np.asarray(pred, np.float64)
    t = np.asarray(true, np.float64)
    if c == "PERM":
        return np.maximum(0.0, 1.0 - np.abs(np.log10(np.maximum(p / np.maximum(t, EPS), EPS))))
    tol = 0.08 if c == "POR" else 0.05
    return np.maximum(0.0, 1.0 - np.abs(p - t) / (tol * (np.abs(t) + EPS)))


def log10ify(c: str, r: np.ndarray) -> np.ndarray:
    r = np.asarray(r, np.float64)
    return r if c == "PERM" else np.log10(np.maximum(r, EPS))


def auc(yt: np.ndarray, p: np.ndarray) -> float:
    """二分类直接算；多分类给 sklearn 1.7 显式指定 multi_class='ovr'（默认已被移除）."""
    p = np.asarray(p, np.float64)
    if p.ndim == 1 or p.shape[1] == 2:
        return float(roc_auc_score(yt.astype(int), p[:, 1] if p.ndim == 2 else p))
    return float(roc_auc_score(yt.astype(int), p, multi_class="ovr"))


def fit_clf(Xtr: np.ndarray, yt: np.ndarray, cw: str, Xva: np.ndarray,
            n_class: int = 2) -> np.ndarray:
    """拟合一个 HistGB 分类器，返回验证集概率后立刻释放模型。"""
    from sklearn.ensemble import HistGradientBoostingClassifier
    p = dict(learning_rate=0.1, max_iter=CLF_ITERS, max_leaf_nodes=31,
             min_samples_leaf=50, l2_regularization=1.0, class_weight=cw)
    m = HistGradientBoostingClassifier(loss="log_loss", random_state=42, **p)
    m.fit(Xtr, yt)
    P = m.predict_proba(Xva)
    del m
    gc.collect()
    # 二分类只返回正类概率；多分类返回完整 (n, k) 矩阵
    return P[:, 1].astype(np.float32) if P.shape[1] == 2 else P.astype(np.float32)


def best_th_acc(pmv: np.ndarray, c: str, regb: np.ndarray, y: np.ndarray
                ) -> tuple[float, float]:
    """验证集内扫阈值，返回 (最优 acc, 最优 th)。入参已按掩码截断。"""
    fd = acc_vec(c, np.full(pmv.shape[0], DEFAULT_VALUES[c]), y)
    fr = acc_vec(c, regb, y)
    order = np.argsort(pmv, kind="stable")
    pos = np.searchsorted(pmv[order], TH_GRID, side="left")
    Fd = np.concatenate([[0.0], np.cumsum(fd[order])])
    Fr = np.concatenate([[0.0], np.cumsum(fr[order])])
    n = len(pmv)
    acc = (Fr[pos] + (Fd[n] - Fd[pos])) / n
    j = int(np.argmax(acc))
    return float(acc[j]), float(TH_GRID[j])


def acc_at(pmv: np.ndarray, c: str, regb: np.ndarray, y: np.ndarray,
           m: np.ndarray, th: float) -> float:
    pred = np.where(pmv >= th, DEFAULT_VALUES[c], regb)
    return float(acc_vec(c, pred[m], y[m]).mean())


def feature_cols() -> list[str]:
    d = pd.DataFrame({"DEPTH": [0.0, 1.0], "GR": [0.0, 0.0], "PE": [0.0, 0.0],
                      "SP": [0.0, 0.0], "CAL": [0.0, 0.0], "AC": [0.0, 0.0],
                      "DEN": [0.0, 0.0], "CNL": [0.0, 0.0], "RXO": [0.0, 0.0],
                      "RT": [0.0, 0.0], "DEVI": [0.0, 0.0], "AZIM": [0.0, 0.0],
                      "BIT": [0.0, 0.0], "logId": ["x", "x"]})
    F.LAG_OFFSETS = list(LAGS)
    F.WIN_SIZES = list(WINS)
    return list(F.make_features(d, context=True, priors=True).columns)


def build(wells: list[pd.DataFrame]) -> dict:
    """特征表算一次存 npz 复用（657K×225 float32 ≈ 591 MB）。"""
    if os.path.exists(FEAT_CACHE):
        z = np.load(FEAT_CACHE, allow_pickle=True)
        print(f"  复用特征缓存 {FEAT_CACHE}")
        ytr = {c: z[c].astype(np.float64) for c in TARGETS}
        yva = {c: z["va_" + c].astype(np.float64) for c in TARGETS}
        return dict(Xtr=z["Xtr"], Xva=z["Xva"], Mtr=z["Mtr"], Mva=z["Mva"],
                    ytr=ytr, yva=yva)

    ids = [w["logId"].iloc[0] for w in wells]
    va_ids = set(ids[N_TRAIN_WELLS:])
    tr = pd.concat([w for w in wells if w["logId"].iloc[0] not in va_ids], ignore_index=True)
    va = pd.concat([w for w in wells if w["logId"].iloc[0] in va_ids], ignore_index=True)
    F.LAG_OFFSETS = list(LAGS)
    F.WIN_SIZES = list(WINS)
    Ytr, Yva = make_targets(tr), make_targets(va)
    Dtr, Dva = default_mask(Ytr), default_mask(Yva)
    Xtr = F.make_features(tr, context=True, priors=True).to_numpy(np.float32)
    del tr
    gc.collect()
    Xva = F.make_features(va, context=True, priors=True).to_numpy(np.float32)
    del va
    gc.collect()
    Mtr = Dtr.to_numpy().astype(np.int8)
    Mva = Dva.to_numpy().astype(np.int8)
    np.savez_compressed(FEAT_CACHE, Xtr=Xtr, Xva=Xva, Mtr=Mtr, Mva=Mva,
                        **{c: Ytr[c].to_numpy(np.float32) for c in TARGETS},
                        **{"va_" + c: Yva[c].to_numpy(np.float32) for c in TARGETS})
    return dict(Xtr=Xtr, Xva=Xva, Mtr=Mtr, Mva=Mva,
                ytr={c: Ytr[c].to_numpy(np.float64) for c in TARGETS},
                yva={c: Yva[c].to_numpy(np.float64) for c in TARGETS})


def fit_regressors(Xtr: np.ndarray, Mtr: np.ndarray, ytr: dict,
                   Xva: np.ndarray, cols: list[str]) -> dict:
    """部署口径回归器：真实段重训 + weight_tol 样本加权（train.py:278-290）。"""
    out = {}
    for c in TARGETS:
        real = (~Mtr[:, TARGETS.index(c)].astype(bool)) & (~np.isnan(ytr[c]))
        yraw = ytr[c][real]
        w = np.abs(np.log10(np.maximum(yraw, EPS))) if c == "PERM" else np.abs(yraw)
        w = np.clip(w, np.percentile(w, 5), np.percentile(w, 95))
        w = (w / w.mean()).astype(np.float32)
        yt = transform_target(c, pd.Series(yraw)).to_numpy()
        t = time.time()
        reg = _reg(monotonic=POR_MONOTONIC if c == "POR" else None, columns=cols)
        reg.fit(Xtr[real], yt, sample_weight=w)
        out[c] = log10ify(c, reg.predict(Xva)).astype(np.float32)
        print(f"    回归器 {c}: {int(real.sum()):,} 点  {time.time()-t:.0f}s  "
              f"rss {rss_gb():.2f} GB   [{time.time()-T0:.0f}s]")
        del reg, w, yt, yraw
        gc.collect()
    return out


def empirical_marginal(M: np.ndarray) -> np.ndarray:
    """从训练数据估 P(def_c | sum=s)，s∈{0,1,2,3}、c∈{POR,PERM,SW} → (3,4) 表。

    比"对称假设 2/3"更准：SW 的默认占比最高，混合态里最不可能非默认。
    """
    s = M.sum(1)
    T = np.zeros((3, 4))
    for c in range(3):
        for k in range(4):
            sel = s == k
            T[c, k] = 0.5 if sel.sum() == 0 else float(M[sel, c].mean())
    return T


def main() -> None:
    jack = "--jack" in sys.argv
    d = build(load_wells("./train", has_labels=True))
    Xtr, Xva = d["Xtr"], d["Xva"]
    Mtr, Mva = d["Mtr"], d["Mva"]
    ytr, yva = d["ytr"], d["yva"]
    bnd = len(Xtr)
    cols = feature_cols()

    Z = np.load(os.path.join(RUNS, "tcn_all80_parts.npz"))
    assert Z["reg_POR"].shape[0] - bnd == len(Xva), (Z["reg_POR"].shape, bnd, len(Xva))
    ens = {c: np.float32(Z[f"reg_{c}"][bnd:]) for c in TARGETS}
    pdft = {c: np.float32(Z[f"pdf_{c}"][bnd:]) for c in TARGETS}

    print("=" * 104)
    print("联合默认标签可分性实验（±10/w21 = 225 特征，v7 融合口径，clf max_iter=%d）" % CLF_ITERS)
    print("=" * 104)
    print(f"训练 {bnd:,} 点 / 验证 {len(Xva):,} 点 / 特征 {Xtr.shape[1]}   [{time.time()-T0:.0f}s]")
    for nm, M in (("训练集", Mtr), ("验证集", Mva)):
        s = M.sum(1)
        u, cnt = np.unique(s, return_counts=True)
        print(f"  {nm} 联合标签 M.sum(1) 占比: " + "  ".join(
            f"{int(k)}={v/len(s):.4f}" for k, v in zip(u, cnt)))
    print("  单目标默认占比: 训练 " + "  ".join(
        f"{c}={Mtr[:, i].mean():.4f}" for i, c in enumerate(TARGETS))
        + "   |  验证 " + "  ".join(f"{c}={Mva[:, i].mean():.4f}" for i, c in enumerate(TARGETS)))
    nan_tr = {c: int(np.isnan(ytr[c]).sum()) for c in TARGETS}
    nan_va = {c: int(np.isnan(yva[c]).sum()) for c in TARGETS}
    print(f"  目标 NaN 行数: 训练 {nan_tr}   验证 {nan_va}   "
          f"(验证 SW 少 {nan_va['SW']:,} 行 = 病态井 c7611b01…6，AUC/acc 必须按掩码算)")
    TM = empirical_marginal(Mtr)
    print("  经验边际 P(def_c | M.sum=s)（行=目标，列=s=0..3）:")
    for i, c in enumerate(TARGETS):
        print(f"    {c:5s} " + "  ".join(f"{v:.4f}" for v in TM[i]))

    S_tr, S_va = Mtr.sum(1), Mva.sum(1)
    mva = {c: ~np.isnan(yva[c]) for c in TARGETS}
    # 联合标签"三目标全默认/全非默认"要求三目标都已知。default_mask 对 NaN 给 False
    # （= 非默认），会把"未知"误标成"非默认"，必须剔除：
    #   训练 6,018 行（0.9%）三目标有 NaN；验证 10,288 行（14.1%，病态井 c7611b01…6 SW 全 NaN）。
    nans = np.stack([np.isnan(ytr[c]) for c in TARGETS])
    m_jt = ~np.any(nans, axis=0)
    nans_va = np.stack([np.isnan(yva[c]) for c in TARGETS])
    m_j = ~np.any(nans_va, axis=0)
    print(f"  三目标均非 NaN 的行数: 训练 {int(m_jt.sum()):,}/{len(m_jt):,}"
          f"（剔除 {int((~m_jt).sum()):,}）  验证 {int(m_j.sum()):,}/{len(m_j):,}"
          f"（剔除 {int((~m_j).sum()):,}）")
    print(f"  验证集 联合标签 s=3 占比: 剔除NaN后 {float((S_va[m_j] == 3).mean()):.4f}"
          f"   含NaN {float((S_va == 3).mean()):.4f}")
    y_all = (S_tr[m_jt] == 3).astype(int)
    y_non = (S_tr[m_jt] == 0).astype(int)
    y4 = S_tr[m_jt].astype(int)
    Xjtr = Xtr[m_jt]
    l_all, l_non, l4 = ((S_va[m_j] == 3).astype(int), (S_va[m_j] == 0).astype(int),
                        S_va[m_j].astype(int))

    # ── [1] 逐目标基线（balanced = 部署口径），含 TCN 概率融合
    print(f"\n[1] 逐目标基线分类器（class_weight=balanced）   [{time.time()-T0:.0f}s]")
    p_tree, p_def, auc_def, auc_tree = {}, {}, {}, {}
    for i, c in enumerate(TARGETS):
        vt = ~np.isnan(ytr[c])
        pt = fit_clf(Xtr[vt], Mtr[vt, i], "balanced", Xva).astype(np.float32)
        p_def[c] = ((1 - V7["clf_tcn_w"][c]) * pt + V7["clf_tcn_w"][c] * pdft[c]).astype(np.float32)
        auc_tree[c] = auc(Mva[:, i][mva[c]], pt[mva[c]])
        auc_def[c] = auc(Mva[:, i][mva[c]], p_def[c][mva[c]])
        print(f"    {c:5s} AUC(树p_def)={auc_tree[c]:.4f}  AUC(p_def∨TCN)={auc_def[c]:.4f}"
              f"   [{time.time()-T0:.0f}s]")
        del pt
        gc.collect()

    # ── [2][3] 联合标签分类器
    # 二分类跑 balanced 与 None（任务要求）；4 分类只跑 balanced —— 每折 fit 约
    # 40-120 s（另一 agent 同时在跑，CPU 争用），要压在 15 min 预算内。
    print(f"\n[2] 联合标签分类器（二分类 balanced/None，4 分类仅 balanced）   "
          f"[{time.time()-T0:.0f}s]")
    jp, P4s = {}, {}
    for cw in ("balanced", None):
        for nm, yt, lv in (("all_def", y_all, l_all), ("all_nondef", y_non, l_non)):
            p = fit_clf(Xjtr, yt, cw, Xva)
            jp[f"{nm}|{cw}"] = p
            print(f"    {nm:10s} cw={str(cw):9s} AUC(联合自身)={auc(lv, p[m_j]):.4f}"
                  f"   [{time.time()-T0:.0f}s]")
    for cw in ("balanced",):
        P4 = fit_clf(Xjtr, y4, cw, Xva, n_class=4)
        P4s[cw] = P4
        ovr = {k: auc(l4[m_j] == k, P4[m_j, k]) for k in range(4)}
        macro = auc(l4[m_j], P4[m_j])
        print(f"    four_class cw={str(cw):9s} 宏AUC(OVR)={macro:.4f}  "
              + "  ".join(f"P(s={k})={v:.4f}" for k, v in ovr.items())
              + f"   [{time.time()-T0:.0f}s]")

    print(f"\n[3] 联合分数 → 各目标「默认概率」代理（对单目标默认标签算 AUC）   "
          f"[{time.time()-T0:.0f}s]")
    proxies: dict[str, dict[str, np.ndarray]] = {}
    for cw in ("balanced", None):
        for nm in ("all_def", "all_nondef"):
            key = f"{nm}|{cw}"
            src = jp[key]
            proxies[key] = {c: (src if nm == "all_def" else 1.0 - src).astype(np.float32)
                            for c in TARGETS}
    for cw in P4s:
        # 4 分类：按经验边际表逐类加权（P(def_c) = Σ_s P(s)·P(def_c|s)）
        key = f"4c_marg|{cw}"
        proxies[key] = {c: (P4s[cw] @ TM[i]).astype(np.float32) for i, c in enumerate(TARGETS)}
    for key in sorted(proxies):
        print(f"  {key:18s} " + "  ".join(
            f"{c}={auc(Mva[:, TARGETS.index(c)][mva[c]], proxies[key][c][mva[c]]):.4f}"
            for c in TARGETS))

    # ── 掩码口径对照（决定结论的关键）
    # 病态井 c7611b01…6 的 SW 全 NaN → default_mask 给 False → 联合标签被强制 s=0，
    # 而模型在这 9,654 行上 p_all 均值 0.5677（预测"像储层"）。按逐目标掩码算
    # 联合 AUC 会**恰好剔除模型最差的 14.1% 行**，得到虚高的 0.9742。
    # 逐目标 AUC（0.8959/0.8917/0.9727）里 POR/PERM 是**含**这些行的，所以必须用
    # 全量口径 0.8934 对比，才是同一批点。
    print(f"\n[3b] 掩码口径对照（同一 p_all，只换标签掩码）   [{time.time()-T0:.0f}s]")
    pa = jp["all_def|balanced"]
    a_full = auc((S_va == 3).astype(int), pa)
    a_mask = auc((S_va[m_j] == 3).astype(int), pa[m_j])
    n_bad = int((~m_j).sum())
    print(f"  AUC(p_all, s==3) 全量口径   = {a_full:.4f}  n={len(pa):,}")
    print(f"  AUC(p_all, s==3) 三目标非NaN = {a_mask:.4f}  n={int(m_j.sum()):,}"
          f"  ← 剔除了模型最差的 {n_bad:,} 行（病态井 SW 全 NaN，标签被强制 s=0）")
    print(f"  逐目标基线 AUC（含这些行，除 SW 外无 NaN）: "
          + "  ".join(f"{c}={auc_def[c]:.4f}" for c in TARGETS))
    print(f"  → 可比口径是 {a_full:.4f}，不是 {a_mask:.4f}；虚高 {a_mask - a_full:+.4f} 是掩码选择性剔除的假象。")

    # ── [3c] λ 扫描的 AUC 部分（不需要回归器，先算 —— 这是判定主证据）
    # 阈值随 λ 重扫属二次选择、会系统性抬高 acc，故把 AUC（阈值无关）提前，
    # 即使后面因时间预算被砍，判定也已经有结论。
    print(f"\n[3c] λ 扫描 AUC（阈值无关，主证据）：AUC( (1-λ)·p_def + λ·p_joint )   "
          f"[{time.time()-T0:.0f}s]")
    print(f"  {'代理':18s}{'目标':5s}{'AUC(p_def)':>11s}{'AUC(p_j)':>10s}{'AUC(λ*)':>10s}"
          f"{'λ*':>6s}{'AUCΔ':>8s}  网格 0/0.25/0.5/0.75/1.0")
    auc_rows = []
    for key in sorted(proxies):
        for c in TARGETS:
            m = mva[c]
            pj = proxies[key][c]
            grid = []
            for lam in LAM_GRID:
                pm = ((1 - lam) * p_def[c] + lam * pj)[m]
                grid.append((lam, auc(Mva[:, TARGETS.index(c)][m], pm)))
            j = int(np.argmax([g[1] for g in grid]))
            lam_s, a_s = grid[j]
            a0, a1 = auc_def[c], grid[-1][1]
            auc_rows.append((key, c, a0, a1, a_s, lam_s))
            print(f"  {key:18s}{c:5s}{a0:11.4f}{a1:10.4f}{a_s:10.4f}{lam_s:6.2f}{a_s-a0:+8.4f}"
                  + "  " + " ".join(f"{g[1]:.4f}" for g in grid[::5]))
    best_auc = max(auc_rows, key=lambda r: r[4] - r[2])
    print(f"\n  最大单目标 AUC 增益: {best_auc[0]} / {best_auc[1]} = {best_auc[4]-best_auc[2]:+.4f}"
          f"  (λ*={best_auc[5]:.2f})")
    print(f"  判定标准：AUC 必须真升才算数。|ΔAUC| < 0.005 视为无独立排序信息。")

    # ── [4] 回归器 + 自检
    print(f"\n[4] 回归器（weight_tol，部署口径）   [{time.time()-T0:.0f}s]")
    rt = fit_regressors(Xtr, Mtr, ytr, Xva, cols)
    regb = {c: np.power(10.0, (1 - V7["blend_w"][c]) * rt[c] + V7["blend_w"][c] * ens[c])
            .astype(np.float64) for c in TARGETS}

    print(f"\n[5] 自检：纯基线（λ=0）   [{time.time()-T0:.0f}s]")
    print(f"  {'目标':5s}{'acc@v7th':>10s}{'acc@最优th':>12s}{'最优th':>9s}{'v7th':>7s}"
          f"{'贡献':>8s}{'acc_real':>10s}{'判默认占比':>12s}")
    base_acc, base_th, base_tot = {}, {}, 0.0
    for c in TARGETS:
        m = mva[c]
        a7 = acc_at(p_def[c], c, regb[c], yva[c], m, V7["thresholds"][c])
        ab, thb = best_th_acc(p_def[c][m], c, regb[c][m], yva[c][m])
        dec = (p_def[c] >= V7["thresholds"][c])[m]
        ad_real = (float(acc_vec(c, regb[c][m], yva[c][m])[~dec].mean())
                   if (~dec).sum() else float("nan"))
        base_acc[c], base_th[c] = ab, thb
        base_tot += WEIGHTS[c] * ab * 100
        print(f"  {c:5s}{a7:10.4f}{ab:12.4f}{thb:9.3f}{V7['thresholds'][c]:7.3f}"
              f"{WEIGHTS[c]*ab*100:8.3f}{ad_real:10.4f}{dec.mean():12.4f}")
    print(f"\n  *** 自检基线 Total = {base_tot:.3f}（部署 v7 = 73.749，容差 ±0.05）***")
    print(f"  判定: {'通过' if abs(base_tot - 73.749) <= 0.05 else '不通过 → harness 有偏，后续 Δ 不可信'}")
    print(f"  注: clf max_iter=150（部署 300），预期 AUC 略低 ~0.001；总分为阈值扫后的读数。")

    # ── [6] λ 融合扫描
    print(f"\n[6] λ 扫描：(1-λ)·p_def + λ·p_joint，每个 λ 在验证集重扫阈值   "
          f"[{time.time()-T0:.0f}s]")
    print(f"  {'代理':18s}{'目标':5s}{'λ*':>5s}{'AUC':>8s}{'AUCΔ':>8s}{'acc':>8s}"
          f"{'th*':>8s}{'@v7th':>8s}{'Δ贡献':>9s}{'Δacc':>9s}")
    rows, best_k, best_tot = [], None, -1.0
    for key in sorted(proxies):
        tot = 0.0
        for c in TARGETS:
            m = mva[c]
            pj = proxies[key][c]
            best = None
            for lam in LAM_GRID:
                pm = ((1 - lam) * p_def[c] + lam * pj).astype(np.float64)
                a_u = auc(Mva[:, TARGETS.index(c)][m], pm[m])
                ab, thb = best_th_acc(pm[m], c, regb[c][m], yva[c][m])
                af = acc_at(pm, c, regb[c], yva[c], m, V7["thresholds"][c])
                if best is None or ab > best[2]:
                    best = (lam, a_u, ab, thb, af)
            lam, a_u, ab, thb, af = best
            tot += WEIGHTS[c] * ab * 100
            rows.append((key, c, lam, a_u, ab, thb, af))
            print(f"  {key:18s}{c:5s}{lam:5.2f}{a_u:8.4f}{a_u-auc_def[c]:+8.4f}"
                  f"{ab:8.4f}{thb:8.3f}{af:8.4f}"
                  f"{WEIGHTS[c]*(ab-base_acc[c])*100:+9.4f}{ab-base_acc[c]:+9.4f}")
        if tot > best_tot:
            best_tot, best_k = tot, key
    print(f"\n  基线 Σ最优 = {base_tot:.3f}   最佳代理 {best_k} Σ最优 = {best_tot:.3f}"
          f"   Δ = {best_tot-base_tot:+.3f}")
    ups = [r[1] for r in rows if r[0] == best_k and r[3] > auc_def[r[1]]]
    print(f"  {best_k} 下 AUC 真升的目标: "
          f"{ups or '无（联合分数没有提供新的排序信息）'}")

    out = dict(baseline=base_tot, rows=rows, best_k=best_k, best_tot=best_tot,
               auc_rows=auc_rows, best_auc=best_auc,
               auc_def=auc_def, auc_tree=auc_tree, base_acc=base_acc, base_th=base_th,
               proxies=sorted(proxies), rss_gb=rss_gb(), elapsed=time.time()-T0)
    if jack:
        out.update(jackknife(d, cols, rt, regb, ens, pdft, proxies, base_tot))
    joblib.dump(out, os.path.join(RUNS, "joint_label_summary.joblib"))
    print(f"\n  峰值 RSS = {rss_gb():.2f} GB（macOS ru_maxrss，高水位）   "
          f"总耗时 {time.time()-T0:.1f} s")
    print("  已保存 experiments/runs/joint_label_summary.joblib")


def jackknife(d: dict, cols: list[str], rt: dict, regb: dict, ens: dict, pdft: dict,
              proxies: dict, base_tot: float) -> dict:
    """留一井复核。8 口井才是实验单元；行级 bootstrap 在本项目无效（72,779 行被 8 井
    聚类，单井内 POR lag-1 自相关 0.967，SE ∝ 1/√N 低估方差约一个数量级）。

    成本控制：逐折只重训**分类器**（逐目标 balanced + 4 个联合代理的 balanced 版），
    回归器与阈值固定用全量值。这是**保守**方向 —— 候选在阈值和回归上拿不到
    拟合优势，仍为正才算数。
    """
    Xtr0, Mtr0 = d["Xtr"], d["Mtr"]
    ytr0 = d["ytr"]
    wells = load_wells("./train", has_labels=True)
    ids = [w["logId"].iloc[0] for w in wells]
    ids_va = ids[N_TRAIN_WELLS:]
    wmap = {w["logId"].iloc[0]: w for w in wells}
    # 验证集是 8 井按 logId 顺序 concat 的，各井长度不等 —— 用行索引区间定位，
    # 不能按 f*len 均分（踩坑点）。
    n_va = len(d["Xva"])
    idx = []
    i0 = 0
    for h in ids_va:
        n = len(wmap[h])
        idx.append((i0, i0 + n))
        i0 += n
    assert i0 == n_va, (i0, n_va)
    print(f"\n[7] 留一井 jackknife（回归器与阈值固定，仅重训分类器）   [{time.time()-T0:.0f}s]")

    acc = {k: np.zeros(len(ids_va)) for k in proxies}
    acc["baseline"] = np.zeros(len(ids_va))
    for f, hid in enumerate(ids_va):
        lo, hi = idx[f]
        Xf = d["Xva"][lo:hi]
        Df = d["Mva"][lo:hi]
        yf = {c: d["yva"][c][lo:hi] for c in TARGETS}
        row = pd.concat([w for k, w in wmap.items() if k != hid], ignore_index=True)
        row_Y = make_targets(row)
        Xw = F.make_features(row, context=True, priors=True).to_numpy(np.float32)
        Mw = default_mask(row_Y).to_numpy().astype(np.int8)
        S = Mw.sum(1)
        TMw = empirical_marginal(Mw)
        yw = {c: row_Y[c].to_numpy(np.float64) for c in TARGETS}
        del row, row_Y
        gc.collect()
        tot_base = 0.0
        for i, c in enumerate(TARGETS):
            vt = ~np.isnan(ytr0[c])
            p = fit_clf(Xw[vt], Mw[vt, i], "balanced", Xf)
            pm = ((1 - V7["clf_tcn_w"][c]) * p + V7["clf_tcn_w"][c]
                  * pdft[c][lo:hi]).astype(np.float32)
            m = ~np.isnan(yf[c])
            tot_base += WEIGHTS[c] * best_th_acc(pm[m], c, regb[c][lo:hi][m], yf[c][m])[0] * 100
            del p, pm
            gc.collect()
        acc["baseline"][f] = tot_base
        # 3 个 balanced 联合代理（None 版在主实验已测，jackknife 只复核 balanced）
        pj_cache = {}
        nans_w = np.stack([np.isnan(yw[c]) for c in TARGETS])
        mk = ~np.any(nans_w, axis=0)
        for tag in ("all_def", "all_nondef"):
            pj_cache[tag] = fit_clf(Xw[mk], ((S[mk] == 3) if tag == "all_def" else (S[mk] == 0)).astype(int),
                                    "balanced", Xf)
        pj_cache["4c_marg"] = fit_clf(Xw[mk], S[mk].astype(int), "balanced", Xf, n_class=4)
        for key in proxies:
            if key.endswith("|None"):
                continue
            tot = 0.0
            pj = pj_cache[key.split("|")[0]]
            for i, c in enumerate(TARGETS):
                pjc = (pj if key.startswith(("all_def", "4c")) else 1.0 - pj)
                if key.startswith("4c_marg"):
                    pjc = pj @ TMw[i]
                pm = ((1 - V7["clf_tcn_w"][c]) * pjc + V7["clf_tcn_w"][c]
                      * pdft[c][lo:hi]).astype(np.float32)
                m = ~np.isnan(yf[c])
                tot += WEIGHTS[c] * best_th_acc(pm[m], c, regb[c][lo:hi][m], yf[c][m])[0] * 100
            acc[key][f] = tot
        del Xw, Xf, Mw, pj_cache
        gc.collect()
        print(f"  折 {f+1}/8  holdout={hid[:10]}  baseline={acc['baseline'][f]:.3f}   "
              f"[{time.time()-T0:.0f}s]  rss {rss_gb():.2f} GB")

    print(f"\n  留一井逐井总分（回归器/阈值固定，仅分类器重训）：")
    print(f"  {'代理':18s}" + "  ".join(f"{h[:6]:>7s}" for h in ids_va) + f"  {'均值':>8s}{'Δ基线':>9s}")
    for k in ["baseline"] + [x for x in sorted(proxies) if not x.endswith("|None")]:
        v = acc[k]
        print(f"  {k:18s}" + "  ".join(f"{x:7.3f}" for x in v)
              + f"  {v.mean():8.3f}{v.mean()-acc['baseline'].mean():+9.3f}")
    per_well_delta = {k: acc[k] - acc["baseline"] for k in proxies if not k.endswith("|None")}
    print(f"\n  逐井 Δ（相对 baseline）与正井数：")
    for k in sorted(per_well_delta):
        dv = per_well_delta[k]
        n = len(dv)
        print(f"  {k:18s} 均值 {dv.mean():+7.3f}  范围 [{dv.min():+6.3f},{dv.max():+6.3f}]"
              f"  正井 {int((dv > 0).sum())}/{n}"
              f"  jackknife[{(n-1)/n*(dv.mean()-dv.min()):+7.3f},"
              f"{(n-1)/n*(dv.mean()-dv.max()):+7.3f}]")
    return dict(per_well={k: acc[k].tolist() for k in acc},
                delta={k: per_well_delta[k].tolist() for k in per_well_delta},
                ids_va=ids_va, rss_gb=rss_gb(), elapsed=time.time()-T0)


if __name__ == "__main__":
    main()
