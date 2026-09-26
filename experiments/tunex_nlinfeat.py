"""跨曲线非线性交互特征实验（2026-09-26）：新特征是否带来新的**排序信息**（AUC）。

背景与定位
----------
CLAUDE.md §4.11 + §4.12 已三次独立确认 AUC 天花板 ≈ 0.897 属于**特征的信息上限**：
  (1) 六种分类器训练方式只让 AUC 动 0.000～0.006；
  (2) 树 AUC(0.8966) ≈ TCN AUC(0.8909)，两路融合提不了 AUC；
  (3) 上下文窗口 ±10/w21 → ±60/w41（225→321 特征）让 POR/PERM 的 AUC **下降**。
§4.2 又证真值分段 + 现有回归器 = 76.336（screen 不可达上界），2.587 分全卡在分段决策；
POR/SW 回归器已到顶，回归加权/段级模型/逐点选择器/融合变体/更多 TCN 全部已否决。

所以本次只试一类东西：**尚未存在于 225 个特征里的量**。基座固定 225 特征（±10/w21），
逐个追加候选特征组，看分段器 AUC 是否移动。

已否决、不重试：上下文窗口加宽（§4.12，AUC 反降）；逐井 pct-rank（§7，全井秩已试）；
物理先验 POR_D/POR_S/POR_avg/POR_ds（已在 src/feature.py，属基座）。

为什么这些候选可能是新的
------------------------
基座 225 列 = 13 原始曲线 + 3 个 RT 变换 + 4 物理先验 + rel_depth + 12 曲线 ×
(10 lag + 3 窗口×(mean,std) + diff1)。基座里**没有**：
  * 长于 w21 的任何窗口统计；
  * 窗口 min/max —— 只有 mean/std，"本点在邻域内有多极端" 从未显式给出；
  * 曲线间乘积（除 RT_over_RXO 与先验里隐含的 DEN/AC/CNL 组合）；
  * 去趋势残差（本点值 − 长时间窗口中位数 = 区域趋势之上的局部异常）；
  * 二阶差分；  * |diff1| 的窗口统计。
HistGB 深度 ~5（max_leaf_nodes=31）能自己表示二曲线的与逻辑，但**跨 81～201 个点聚合
成单点特征**（长窗口中位数、min/max）没有任何现有单点列能替代 —— 那是真新的信息。

内存纪律（本机 16 GB，可用 ~6.2 GB，可能另一 agent 同时在跑）
--------------------------------------------------------------
全程 float32；每个 fit 完立刻 `del obj; gc.collect()`；特征构造一次；同一时刻不持有
多个 (N,k) 大矩阵；HistGB 回归 max_iter 不上调（用部署的 400，分类器筛选阶段降到 100）。
峰值 RSS 用 resource 记录并在报告里给出。sklearn-only，无 torch，不 import run_tcn*。

用法: /opt/anaconda3/bin/python experiments/tunex_nlinfeat.py
"""
from __future__ import annotations

import gc
import json
import os
import resource
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src.feature as F  # noqa: E402
from src.feature import (  # noqa: E402
    CURVE_COLS,
    default_mask,
    load_wells,
    make_targets,
    transform_target,
)
from src.model import POR_MONOTONIC, _clf, _reg  # noqa: E402
from src.utils import DEFAULT_VALUES, EPS, TARGETS, WEIGHTS  # noqa: E402

RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
OUT_JSON = os.path.join(RUNS, "nlinfeat.json")
N_TRAIN_WELLS = 72
BND = 657489
RAW13 = ["DEPTH", "GR", "PE", "SP", "CAL", "AC", "DEN", "CNL",
         "RXO", "RT", "DEVI", "AZIM", "BIT"]
LAGS = [-10, -5, -3, -2, -1, 1, 2, 3, 5, 10]   # 基座 225 特征口径
WINS = [5, 11, 21]
V7 = {"blend_w": {"POR": 0.50, "PERM": 0.50, "SW": 0.75},
      "thresholds": {"POR": 0.13, "PERM": 0.27, "SW": 0.05},
      "clf_tcn_w": {"POR": 0.60, "PERM": 0.70, "SW": 0.30}}
DEPLOY_V7 = 73.749
TH_GRID = np.arange(0.01, 0.991, 0.01)
SCREEN_CLF = {"max_iter": 60}   # 快速筛选：同一配置跑 base 与所有候选 → 只看 ΔAUC
GATE = 0.002                     # ΔAUC 门槛，低于此不进完整口径验证
T0 = time.time()
PEAK = [0.0]


def rss_gb() -> float:
    v = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9
    if v > PEAK[0]:
        PEAK[0] = v
    return PEAK[0]


def say(s: str) -> None:
    print(s, flush=True)


def acc_vec(c: str, pred, true) -> np.ndarray:
    """官方口径逐点 acc（返回数组，便于按井分解）。"""
    p = np.asarray(pred, np.float64)
    t = np.asarray(true, np.float64)
    if c == "PERM":
        return np.maximum(0.0, 1.0 - np.abs(np.log10(np.maximum(p / np.maximum(t, EPS), EPS))))
    tol = 0.08 if c == "POR" else 0.05
    return np.maximum(0.0, 1.0 - np.abs(p - t) / (tol * (np.abs(t) + EPS)))


def log10ify(c: str, r) -> np.ndarray:
    r = np.asarray(r, np.float64)
    return r if c == "PERM" else np.log10(np.maximum(r, EPS))


def acc_at_th(pm_tree, c, rt_log, reg_tcn, pdf_tcn, y, m, th, cw, bw) -> float:
    pmv = (1 - cw) * pm_tree + cw * pdf_tcn
    regb = np.power(10.0, (1 - bw) * rt_log + bw * reg_tcn)
    pred = np.where(pmv >= th, DEFAULT_VALUES[c], regb)
    return float(acc_vec(c, pred[m], y[m]).mean())


def sweep_best(pm_tree, c, rt_log, reg_tcn, pdf_tcn, y, m, cw, bw):
    """固定 blend_w/clf_tcn_w，只扫阈值，返回 (最优 acc, 最优 th)。"""
    pmv = ((1 - cw) * pm_tree + cw * pdf_tcn)[m]
    regb = np.power(10.0, (1 - bw) * rt_log + bw * reg_tcn)[m]
    fd = acc_vec(c, np.full(int(m.sum()), DEFAULT_VALUES[c]), y[m])
    fr = acc_vec(c, regb, y[m])
    order = np.argsort(pmv, kind="stable")
    pos = np.searchsorted(pmv[order], TH_GRID, side="left")
    Fd = np.concatenate([[0.0], np.cumsum(fd[order])])
    Fr = np.concatenate([[0.0], np.cumsum(fr[order])])
    n = len(pmv)
    acc = (Fr[pos] + (Fd[n] - Fd[pos])) / n
    j = int(np.argmax(acc))
    return float(acc[j]), float(TH_GRID[j])


def well_scores(pm_tree, c, rt_log, reg_tcn, pdf_tcn, y, m, th, cw, bw, well_ids):
    """逐井**得分贡献**（分）。三目标相加 == 官方 Total（逐点等权）。必须带 ×100。"""
    pmv = (1 - cw) * pm_tree + cw * pdf_tcn
    regb = np.power(10.0, (1 - bw) * rt_log + bw * reg_tcn)
    pred = np.where(pmv >= th, DEFAULT_VALUES[c], regb)
    row = np.nan_to_num(100.0 * WEIGHTS[c] * acc_vec(c, pred, y))
    n_valid = m.astype(int)
    n_tot = float(n_valid.sum())
    return pd.Series(np.asarray(row, np.float64) * n_valid / n_tot,
                     index=well_ids).groupby(level=0).sum()


def auc_loo(joined_prob: np.ndarray, labels: np.ndarray, well_ids: np.ndarray) -> tuple:
    """按井拆分的 AUC，再做留一井 jackknife（8 井是实验单元）。返回 (min, max)。"""
    vals = {}
    for w in dict.fromkeys(well_ids):
        iw = well_ids == w
        vals[w] = float(roc_auc_score(labels[iw], joined_prob[iw]))
    jk = {w: float(np.mean([vals[x] for x in vals if x != w])) for w in vals}
    return min(jk.values()), max(jk.values())


# ---------------------------------------------------------------------------
# 候选特征组（只依赖 13 条清洗曲线，推理可用）
# ---------------------------------------------------------------------------
def _by_well(raw, g, fn):
    """对每条曲线在单井内应用 fn（不跨井），返回 12 列 DataFrame。"""
    return pd.DataFrame({c: raw[c].groupby(g, group_keys=False).apply(fn) for c in CURVE_COLS})


def _mm(raw, g, w):
    """局部极值残差：本点 − 邻域最小、邻域最大 − 本点。"""
    out = {}
    for c in CURVE_COLS:
        s = raw[c]
        mn = s.groupby(g, group_keys=False).transform(
            lambda s: s.rolling(w, center=True, min_periods=1).min())
        mx = s.groupby(g, group_keys=False).transform(
            lambda s: s.rolling(w, center=True, min_periods=1).max())
        out[f"lo{w}_{c}"] = (s - mn).to_numpy()
        out[f"hi{w}_{c}"] = (mx - s).to_numpy()
    return pd.DataFrame(out)


def _lrank(s: pd.Series, n: int) -> pd.Series:
    """局部秩：本点在 ±n 邻域（2n+1 点）内的分位。注意不是全井 pct-rank。"""
    cnt = None
    for k in range(-n, n + 1):
        term = (s.shift(k) <= s).astype(np.float32)
        cnt = term if cnt is None else cnt + term
    return cnt / (2 * n + 1)


def build_additions(raw: pd.DataFrame, g: np.ndarray) -> dict:
    out = {}
    # 1) 去趋势残差：本点 − 单井内长窗口中位数（基座最长窗口只有 w21）
    out["det81"] = _by_well(raw, g,
                            lambda s: s - s.rolling(81, center=True, min_periods=1).median())
    out["det201"] = _by_well(raw, g,
                             lambda s: s - s.rolling(201, center=True, min_periods=1).median())
    # 2) 局部极值残差（基座只有 mean/std，没有 min/max）
    out["locminmax51"] = _mm(raw, g, 51)
    out["locminmax101"] = _mm(raw, g, 101)
    # 3) 局部秩 ±10（储层诊断曲线）
    diag = ["DEN", "AC", "CNL", "GR", "PE"]
    out["lrank10"] = pd.DataFrame(
        {f"lrank10_{c}": raw[c].groupby(g, group_keys=False).apply(lambda s: _lrank(s, 10))
         for c in diag})
    # 4) 跨曲线乘积
    CNL, DEN, AC, GR, PE, SP, RT = (raw[c] for c in ("CNL", "DEN", "AC", "GR", "PE", "SP", "RT"))
    rt_rxo = raw["RT"] / raw["RXO"].clip(lower=EPS)
    out["crossprod"] = pd.DataFrame({
        "GRxPE": GR * PE, "DENxAC": DEN * AC, "SPxRT": SP * RT, "DENxCNL": DEN * CNL,
        "GRxCNL": GR * CNL, "PExCNL": PE * CNL, "ACxCNL": AC * CNL,
        "CNLxlogRT": CNL * np.log10(RT.clip(lower=EPS)),
        "DENxRT_RXO": DEN * rt_rxo, "SPxPE": SP * PE})
    # 5) 物理残差：不同测井方法给出的孔隙度互相核对
    pord = (2.65 - DEN) / 1.65 * 100.0
    pors = (AC - 55.5) / (189.0 - 55.5) * 100.0
    out["physresid"] = pd.DataFrame({
        "CNL_minus_PORD": CNL - pord,
        "CNL_minus_PORS": CNL - pors,
        "PORD_minus_PORS": pord - pors,
        "AC_minus_PORDline": AC - (55.5 + (189.0 - 55.5) * pord / 100.0),
        "CNL_times_RT_RXO": CNL * rt_rxo,
        "GR_times_CNL_over_DEN": GR * CNL / DEN.clip(lower=EPS),
        "SP_plus_GR_minus_CNLeq": SP + GR - CNL})
    # 6) 联合指示器：低 DEN 且高 AC 且高 CNL 的与逻辑（逐井 z 分数乘积）
    def wz(col, sign):
        s = raw[col]
        mu = s.groupby(g, group_keys=False).transform("mean")
        sd = s.groupby(g, group_keys=False).transform("std").replace(0.0, 1.0)
        return ((s - mu) / sd) * sign
    zd, za, zc = wz("DEN", -1), wz("AC", 1), wz("CNL", 1)
    out["jointind"] = pd.DataFrame({
        "res_and3": zd * za * zc, "res_pair_DA": zd * za, "res_pair_DC": zd * zc,
        "res_pair_AC": za * zc, "res_min3": np.minimum(zd, np.minimum(za, zc))})
    # 7) 二阶差分 + |diff1| 的窗口统计（基座只有 diff1 原值）
    d = {}
    for c in CURVE_COLS:
        s1 = raw[c].diff(1)
        d[f"diff2_{c}"] = s1.diff(1)
        d[f"adiff1w21_{c}"] = s1.abs().rolling(21, center=True, min_periods=1).mean()
    out["diff2"] = pd.DataFrame(d)
    return {k: v.astype(np.float32) for k, v in out.items()}


CAND_ORDER = ["det81", "locminmax51", "crossprod", "physresid", "det201",
              "locminmax101", "lrank10", "jointind", "diff2"]


def fit_clf_pred(Xtr, Xva, v, Dtr, c, params):
    clf = _clf(params)
    clf.fit(Xtr.loc[v], Dtr.loc[v, c].astype(int).to_numpy())
    p = clf.predict_proba(Xva)[:, 1].astype(np.float32)
    del clf
    gc.collect()
    return p


def fit_reg_pred(Xtr, Xva, real, Ytr, c):
    """真实段回归器，带 weight_tol（train.py:278-290 部署口径，不能关）。"""
    reg = _reg(monotonic=POR_MONOTONIC if c == "POR" else None, columns=list(Xtr.columns))
    yt = transform_target(c, Ytr[c].loc[real]).to_numpy()
    yraw = Ytr[c].loc[real].to_numpy().astype(np.float64)
    w = np.abs(np.log10(np.maximum(yraw, EPS))) if c == "PERM" else np.abs(yraw)
    w = np.clip(w, np.percentile(w, 5), np.percentile(w, 95))
    w = (w / w.mean()).astype(np.float32)
    reg.fit(Xtr.loc[real], yt, sample_weight=w)
    r = log10ify(c, reg.predict(Xva)).astype(np.float32)
    del reg
    gc.collect()
    return r


def main() -> None:
    F.LAG_OFFSETS, F.WIN_SIZES = list(LAGS), list(WINS)
    wells = load_wells("./train", has_labels=True)
    ids = [w["logId"].iloc[0] for w in wells]
    va_ids = set(ids[N_TRAIN_WELLS:])
    tr = pd.concat([w for w in wells if w["logId"].iloc[0] not in va_ids], ignore_index=True)
    va = pd.concat([w for w in wells if w["logId"].iloc[0] in va_ids], ignore_index=True)
    assert len(tr) == BND and len(va) == 72779, (len(tr), len(va))
    yid = va["logId"].to_numpy()
    Ytr, Yva = make_targets(tr), make_targets(va)
    Dtr, Dva = default_mask(Ytr), default_mask(Yva)
    ytr = {c: Ytr[c].to_numpy() for c in TARGETS}
    yva = {c: Yva[c].to_numpy() for c in TARGETS}
    m_va = {c: ~np.isnan(yva[c]) for c in TARGETS}

    Z = np.load(os.path.join(RUNS, "tcn_all80_parts.npz"))
    bnd = len(tr)
    ens = {c: np.float32(Z[f"reg_{c}"][bnd:]) for c in TARGETS}
    pdf = {c: np.float32(Z[f"pdf_{c}"][bnd:]) for c in TARGETS}

    say("=" * 104)
    say("跨曲线非线性交互特征实验（基座 225 特征 ±10/w21；blend_w/clf_tcn_w 固定 v7；阈值验证集扫）")
    say("=" * 104)
    say(f"训练 {len(tr):,} 点 / 验证 {len(va):,} 点  [{time.time()-T0:.0f}s]")

    Xtr = F.make_features(tr, context=True, priors=True)
    Xva = F.make_features(va, context=True, priors=True)
    assert Xtr.shape[1] == 225, Xtr.shape
    recs: list[dict] = []

    # ── 阶段 0：自检（完整口径 clf300 + weight_tol 回归 + v7 阈值）
    say("\n" + "─" * 104)
    say("阶段 0：自检（完整 clf max_iter=300 + weight_tol 回归 + v7 阈值）")
    say("─" * 104)
    self_tot, self_acc, self_auc, self_p, self_r = 0.0, {}, {}, {}, {}
    for c in TARGETS:
        v = ~np.isnan(ytr[c])
        self_p[c] = fit_clf_pred(Xtr, Xva, v, Dtr, c, None)
        self_r[c] = fit_reg_pred(Xtr, Xva, v & (~Dtr[c].to_numpy().astype(bool)), Ytr, c)
        m = m_va[c]
        self_auc[c] = float(roc_auc_score(Dva[c].to_numpy().astype(int)[m], self_p[c][m]))
        a7 = acc_at_th(self_p[c], c, self_r[c], ens[c], pdf[c], yva[c], m,
                       V7["thresholds"][c], V7["clf_tcn_w"][c], V7["blend_w"][c])
        self_acc[c] = a7
        self_tot += WEIGHTS[c] * a7 * 100
    ok = abs(self_tot - DEPLOY_V7) < 0.05
    say(f"自检 screen total = {self_tot:.3f}  vs 部署 v7 = {DEPLOY_V7}"
        f"  (差 {self_tot - DEPLOY_V7:+.3f})  →  {'通过' if ok else '失败，后面所有 Δ 不可信'}")
    say(f"自检 AUC  POR {self_auc['POR']:.4f} / PERM {self_auc['PERM']:.4f} / SW {self_auc['SW']:.4f}"
        f"    (已知参考 0.8966 / 0.8906 / 0.9701)")
    say(f"自检 acc  POR {self_acc['POR']:.4f} / PERM {self_acc['PERM']:.4f} / SW {self_acc['SW']:.4f}"
        f"  [{time.time()-T0:.0f}s]")
    recs.append({"name": "BASE ±10/w21 (selfcheck clf300)", "ncol": 225, "clf": "clf300",
                 "total": self_tot, "por_acc": self_acc["POR"], "perm_acc": self_acc["PERM"],
                 "sw_acc": self_acc["SW"], "por_auc": self_auc["POR"],
                 "perm_auc": self_auc["PERM"], "sw_auc": self_auc["SW"]})
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(recs, f, ensure_ascii=False, indent=1)
    if not ok:
        say("自检未通过 → 终止（harness 有偏，Δ 不可信）。")
        return

    # 自检通过后再构造候选特征（构造会把 RSS 推高，压缩页会让 clf.fit 慢数倍）
    say("\n构造候选特征组（自检已完成，self_p/self_r 保留给阶段 2 复用）...")
    raw_tr, raw_va = Xtr[RAW13].copy(), Xva[RAW13].copy()
    adds_tr = build_additions(raw_tr, tr["logId"].to_numpy())
    del raw_tr
    gc.collect()
    adds_va = build_additions(raw_va, va["logId"].to_numpy())
    del raw_va
    gc.collect()
    for k in CAND_ORDER:
        say(f"  {k:14s} +{adds_tr[k].shape[1]:3d} 列  →  {225 + adds_tr[k].shape[1]:3d} 特征")
    say(f"特征构造完成  [{time.time()-T0:.0f}s]  峰值 RSS {rss_gb()*1000:.0f} MB")

    # ── 阶段 1：快速 AUC 筛选（clf100，只算 AUC）
    say("\n" + "─" * 104)
    say(f"阶段 1：快速筛选（clf max_iter={SCREEN_CLF["max_iter"]}，base 与候选同配置 → 只看 ΔAUC）")
    say("─" * 104)
    fb = {c: fit_clf_pred(Xtr, Xva, ~np.isnan(ytr[c]), Dtr, c, SCREEN_CLF) for c in TARGETS}
    fb_auc = {c: float(roc_auc_score(Dva[c].to_numpy().astype(int)[m_va[c]], fb[c][m_va[c]]))
              for c in TARGETS}
    say(f"  base(fast)    POR {fb_auc['POR']:.4f} / PERM {fb_auc['PERM']:.4f} / SW {fb_auc['SW']:.4f}"
        f"  [{time.time()-T0:.0f}s]")
    for c in TARGETS:
        del fb[c]
    gc.collect()

    fast = []
    for name in CAND_ORDER:
        Xtr_c = pd.concat([Xtr, adds_tr[name]], axis=1)
        Xva_c = pd.concat([Xva, adds_va[name]], axis=1)
        row = {"name": name, "ncol": int(Xtr_c.shape[1]), "clf": "clf100"}
        for c in TARGETS:
            p = fit_clf_pred(Xtr_c, Xva_c, ~np.isnan(ytr[c]), Dtr, c, SCREEN_CLF)
            a = float(roc_auc_score(Dva[c].to_numpy().astype(int)[m_va[c]], p[m_va[c]]))
            row[f"{c}_auc"] = a
            row[f"{c}_auc_delta"] = a - fb_auc[c]
            say(f"  {name:14s} {c:5s} AUC {a:.4f} (Δ {a - fb_auc[c]:+.4f})  [{time.time()-T0:.0f}s]")
            del p
            gc.collect()
        row["max_auc_delta"] = max(row[f"{c}_auc_delta"] for c in TARGETS)
        fast.append(row)
        recs.append(row)
        with open(OUT_JSON, "w", encoding="utf-8") as f:
            json.dump(recs, f, ensure_ascii=False, indent=1)
        del Xtr_c, Xva_c
        gc.collect()

    best = max(fast, key=lambda r: r["max_auc_delta"])
    say(f"\n阶段 1 最优 = {best['name']} (+{best['ncol']-225} 列)，最大 ΔAUC {best['max_auc_delta']:+.4f}")
    say(f"峰值 RSS {rss_gb():.2f} GB  [{time.time()-T0:.0f}s]")

    # ── 阶段 2：完整口径验证（clf300 + reg + 阈值扫 + 逐井 Δ + 留一 jackknife）
    top = [r for r in fast if r["max_auc_delta"] >= GATE]
    say("\n" + "─" * 104)
    say(f"阶段 2：完整口径验证（clf300 + weight_tol 回归 + 阈值扫 + 逐井 Δ + 留一井 jackknife）")
    say(f"  ΔAUC ≥ {GATE} 通过：{[r['name'] for r in top]}")
    say("─" * 104)

    # 阶段 0 的 self_p / self_r 就是完整口径（clf300 + weight_tol）的基座预测，直接复用
    bp, br = self_p, self_r
    base_sc, base_th = {}, {}
    for c in TARGETS:
        base_th[c] = V7["thresholds"][c]
        base_sc[c] = well_scores(bp[c], c, br[c], ens[c], pdf[c], yva[c], m_va[c],
                                 base_th[c], V7["clf_tcn_w"][c], V7["blend_w"][c], yid)
    base_tot = float(sum(base_sc.values()).sum())
    say(f"完整口径基线（v7 阈值）= {base_tot:.3f}  [{time.time()-T0:.0f}s]")

    budget_left = 1500.0 - (time.time() - T0)
    n_afford = max(1, int(budget_left // 250))
    if len(top) > n_afford:
        say(f"  时间预算（剩 {budget_left:.0f}s，单候选完整口径约 245s）→ 只验证前 {n_afford} 个")
        top = top[:n_afford]

    for row in top:
        name = row["name"]
        Xtr_c = pd.concat([Xtr, adds_tr[name]], axis=1)
        Xva_c = pd.concat([Xva, adds_va[name]], axis=1)
        cth, csc, cauc, cacc, cp, cr = {}, {}, {}, {}, {}, {}
        for c in TARGETS:
            v = ~np.isnan(ytr[c])
            cp[c] = fit_clf_pred(Xtr_c, Xva_c, v, Dtr, c, None)
            # 候选回归器在**新特征集**上重拟合（weight_tol），不能复用基座回归器
            cr[c] = fit_reg_pred(Xtr_c, Xva_c, v & (~Dtr[c].to_numpy().astype(bool)), Ytr, c)
            cacc[c] = float(acc_at_th(cp[c], c, cr[c], ens[c], pdf[c], yva[c], m_va[c],
                                      V7["thresholds"][c], V7["clf_tcn_w"][c], V7["blend_w"][c]))
            cauc[c] = float(roc_auc_score(Dva[c].to_numpy().astype(int)[m_va[c]], cp[c][m_va[c]]))
            a, th = sweep_best(cp[c], c, cr[c], ens[c], pdf[c], yva[c], m_va[c],
                               V7["clf_tcn_w"][c], V7["blend_w"][c])
            csc[c] = well_scores(cp[c], c, cr[c], ens[c], pdf[c], yva[c], m_va[c],
                                 th, V7["clf_tcn_w"][c], V7["blend_w"][c], yid)
            cth[c] = th
            say(f"  {name:14s} {c:5s} AUC {cauc[c]:.4f} (Δvs自检 {cauc[c]-self_auc[c]:+.4f}) "
                f"acc@最优th {a:.4f} th {th:.3f}(基线{V7['thresholds'][c]:.3f})  [{time.time()-T0:.0f}s]")
        cand_tot = float(sum(csc.values()).sum())
        d = cand_tot - base_tot
        dw = {k: float(csc[k].sum() - base_sc[k].sum()) for k in base_sc["POR"].index}
        lo, hi = d - max(dw.values()), d - min(dw.values())
        pos = sum(1 for v in dw.values() if v > 0)
        rec = {"name": f"{name} [full clf300]", "ncol": row["ncol"], "clf": "clf300",
               "total": cand_tot, "delta": d,
               "por_auc": cauc["POR"], "perm_auc": cauc["PERM"], "sw_auc": cauc["SW"],
               "por_acc": cacc["POR"], "perm_acc": cacc["PERM"], "sw_acc": cacc["SW"],
               "por_acc_best": cacc.get("POR"), "loo_total": [lo, hi],
               "wells_positive": pos, "wells": len(dw), "per_well_delta": dw}
        rec["auc_loo_jackknife"] = {c: list(auc_loo(cp[c], Dva[c].to_numpy().astype(int), yid))
                                    for c in TARGETS}
        recs.append(rec)
        with open(OUT_JSON, "w", encoding="utf-8") as f:
            json.dump(recs, f, ensure_ascii=False, indent=1)
        say(f"  → Total {cand_tot:.3f} (Δ {d:+.3f})；逐井正号 {pos}/{len(dw)}；"
            f"留一井 jackknife [{lo:+.3f}, {hi:+.3f}] {'跨 0' if lo <= 0 <= hi else '不跨 0'}")
        for c in TARGETS:
            del cp[c]
        gc.collect()
        del Xtr_c, Xva_c
        gc.collect()

    # ── 汇总
    say("\n" + "=" * 104)
    say("汇总（screen 单切分，官方口径；部署 v7 = 73.749）")
    say("=" * 104)
    say(f"{'候选':22s}{'列':>6s}{'clf':>8s}{'AUC POR':>10s}{'PERM':>9s}{'SW':>9s}{'max ΔAUC':>10s}")
    for r in recs:
        if r["clf"] == "clf100":
            say(f"{r['name'][:21]:22s}{r['ncol']:6d}{r['clf']:>8s}"
                f"{r['POR_auc']:10.4f}{r['PERM_auc']:9.4f}{r['SW_auc']:9.4f}"
                f"{r['max_auc_delta']:+10.4f}")
    say(f"{'(fast base)':22s}{225:6d}{'clf100':>8s}"
        f"{fb_auc['POR']:10.4f}{fb_auc['PERM']:9.4f}{fb_auc['SW']:9.4f}{0.0:+10.4f}")
    say("\n完整口径（clf300，回归用基座回归器 + 阈值在验证集扫）：")
    say(f"{'候选':22s}{'列':>6s}{'AUC POR':>10s}{'PERM':>9s}{'SW':>9s}{'Total':>9s}{'Δ':>9s}{'正井':>7s}{'留一井 jackknife':>22s}")
    for r in recs:
        if r["clf"] == "clf300":
            if "loo_total" in r:
                say(f"{r['name'][:21]:22s}{r['ncol']:6d}{r['por_auc']:10.4f}{r['perm_auc']:9.4f}"
                    f"{r['sw_auc']:9.4f}{r['total']:9.3f}{r['delta']:+9.3f}"
                    f"{r['wells_positive']}/{r['wells']:>4s}"
                    f"{r['loo_total'][0]:+.3f}..{r['loo_total'][1]:+.3f}")
            else:
                say(f"{r['name'][:21]:22s}{r['ncol']:6d}{r['por_auc']:10.4f}{r['perm_auc']:9.4f}"
                    f"{r['sw_auc']:9.4f}{r['total']:9.3f}{'':>9s}")
    say(f"\n峰值 RSS {rss_gb():.2f} GB（本机 16 GB，可用 ~6.2 GB）")
    say(f"结果写入 {OUT_JSON}")
    say(f"总耗时 {time.time()-T0:.1f}s")


if __name__ == "__main__":
    main()
