"""逐井尺度归一化 / 去趋势实验（2026-09-26）。

动机
----
训练 72 井、验证另外 8 井，两个集合**不同源**（screen 默认段占比比训练集低 7–9 pp，
且含病态井 c7611b0148bb4b）。部署的 HistGB 树是全局拟合：如果每口井的曲线有各自的整体
水平/斜率偏移，全局模型必须自己把"井与井之间的尺度差"吸收掉，才能把 8 口未见井分对。
官方分数正是按测试集 10 口井的分布算的，所以这是正面攻击真实风险。

部署现状：`scaler.pkl` 只标准化 TCN 输入（tcn_medians/tcn_stds），**树通路完全没有任何
归一化**（HistGB 逐特征尺度不变，全局模型也不需要）。所以"给树通路加逐井尺度信息"是
真正未试过的方向，不是重复 §4.12 的上下文窗口实验。

测什么（基座固定 225 特征 ±10/w21，v7 融合配置；每个变体用完整 TwoStageModel 口径：
clf300 分段器 + reg400 真实段回归 + weight_tol，见 §4.10）
--------------------------------------------------------
  C1 wellz       12 条曲线**替换**为本井 z 分数（只用该井自身均值/标准差 → 推理无泄漏）
  C2 wellmm      12 条曲线**替换**为本井 min-max
  C3 +wellz      基座 + 12 条本井 z 分数（原曲线 + 归一化曲线双份信息）
  C4 +det201m    基座 + 12 条 (x − 本井 w=201 滚动均值低频包络)
  C5 +depth      基座 + rel_depth01（本井深度归一化 [0,1]）+ 深度分箱 dbin10 / dbin20

归一化**全部按 logId 分组**，绝不跨井混算统计量；推理时同样只用该井自身统计量，所以
无泄漏。DEPTH 保持绝对值不缩放（`FEATURES` 里本来就有绝对 DEPTH，地层身份不丢）。

自检
----
阶段 0a：screen_tree_cache（v7 已部署树的 8 井预测）→ 必须 == 73.749，校验打分口径。
阶段 0b：本管线重拟合基座 → 必须落 73.749 ± 0.05（已知 harness 偏高约 +0.013，
          EXPERIMENTS.md §8.16.3 的 POR 7.1e-3 log10 残留，属正常）。
阶段 1：clf60 快速 AUC 筛选（base 与候选同配置 → 只看 ΔAUC）。
阶段 2：完整口径验证（clf300 + reg400 + 阈值扫 + 逐井 Δ + 留一井 jackknife）。

8 口井才是实验单元：行级 bootstrap 在本项目无效（72,779 行被 8 口井聚类，POR 单井
lag-1 自相关 0.967，方差低估约一个数量级）。留一井 jackknife 用
[Δ − max(逐井Δ), Δ − min(逐井Δ)]，均值 = 7/8·Δ。

内存纪律（16 GB，可能另一 agent 同时在跑）：全程 float32；每个 fit 完 `del obj;
gc.collect()`；特征表算一次落盘 npy 复用；同一时刻只持有一个大矩阵；clf 60/300、
reg 400（部署口径）不上调。sklearn-only，无 torch，不 import experiments/run_tcn*。

用法: /opt/anaconda3/bin/python experiments/tunex_wellnorm.py
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

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src.feature as F  # noqa: E402
from src.feature import (CURVE_COLS, FEATURES, default_mask, load_wells,  # noqa: E402
                         make_targets, transform_target)
from src.model import POR_MONOTONIC, _clf, _reg  # noqa: E402
from src.utils import DEFAULT_VALUES, EPS, TARGETS, WEIGHTS, total_score  # noqa: E402

RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
FEATDIR = os.path.join(RUNS, "wellnorm_feat")
OUT_JSON = os.path.join(RUNS, "wellnorm.json")
N_TRAIN_WELLS = 72
BND = 657489
N_VA = 72779
# 基座上下文窗口。默认 ±10/w21 = 225 特征（v7 基座）；
# 用 --ctx v8 切到 ±60/w41 = 321 特征（v8 基座），回答「wellz 增益是否依赖窗口宽度」。
CTX_PRESETS = {
    "v7": ([-10, -5, -3, -2, -1, 1, 2, 3, 5, 10], [5, 11, 21]),
    "v8": ([-60, -30, -20, -10, -5, -3, -2, -1, 1, 2, 3, 5, 10, 20, 30, 60],
           [5, 11, 21, 41]),
}
LAGS, WINS = CTX_PRESETS["v7"]
V7 = {"blend_w": {"POR": 0.50, "PERM": 0.50, "SW": 0.75},
      "thresholds": {"POR": 0.13, "PERM": 0.27, "SW": 0.05},
      "clf_tcn_w": {"POR": 0.60, "PERM": 0.70, "SW": 0.30}}
DEPLOY_V7 = 73.749
# 每个基座的**已知 harness 自评分**：v7 基座 73.749（部署即它），
# v8 基座 73.832（§4.13 harness 72 井估计）。用于阶段 0b 的锚点偏差判断，
# 不是硬门槛 —— Δ 始终相对**本次重拟合的基座**算，锚点只是口径自检的参照。
ANCHOR = {"v7": 73.749, "v8": 73.832}
TH_GRID = np.arange(0.01, 0.991, 0.01)
SCREEN_CLF = {"max_iter": 60}
GATE = 0.0015           # ΔAUC 门槛（历史实测：换训练方式只让 AUC 动 0.000～0.006）
PATHO = "c7611b0148bb4b878c00bc6d1367a136"   # 病态井：POR 默认占比 0.001、SW 全 NaN
CANDS = [("wellz", "replace", None), ("wellmm", "replace", None),
         ("plus_wellz", "add", "wellz"), ("plus_det201m", "add", "det201m"),
         ("plus_depth", "add", "depth")]
T0 = time.time()
PEAK = [0.0]
REC: list[dict] = []
ST: dict = {}


def rss_gb() -> float:
    v = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9
    if v > PEAK[0]:
        PEAK[0] = v
    return PEAK[0]


def say(s: str = "") -> None:
    print(s, flush=True)


def secs() -> float:
    return time.time() - T0


def dump() -> None:
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump({"deploy_v7": DEPLOY_V7, "harness_base": ST.get("base_tot"),
                   "va_wells": ST.get("va_ids"), "recs": REC}, f, ensure_ascii=False, indent=1)


# ---------------------------------------------------------------------------
# 官方口径打分（与 src.utils 同式，数组版以便逐井分解）
# ---------------------------------------------------------------------------
def acc_vec(c: str, pred, true) -> np.ndarray:
    p = np.asarray(pred, np.float64)
    t = np.asarray(true, np.float64)
    if c == "PERM":                       # 恒 1.0 decade 比值容差，无系数无分母
        return np.maximum(0.0, 1.0 - np.abs(np.log10(np.maximum(p / np.maximum(t, EPS), EPS))))
    tol = 0.08 if c == "POR" else 0.05
    return np.maximum(0.0, 1.0 - np.abs(p - t) / (tol * (np.abs(t) + EPS)))


def log10ify(c: str, r) -> np.ndarray:
    r = np.asarray(r, np.float64)
    return r if c == "PERM" else np.log10(np.maximum(r, EPS))


def preds_at(pp, rt, reg_tcn, pdf_tcn, y, c, th, cw, bw) -> np.ndarray:
    """th / cw / bw 均为逐目标 dict。"""
    pmv = (1 - cw[c]) * pp[c] + cw[c] * pdf_tcn[c]
    regb = np.power(10.0, (1 - bw[c]) * rt[c] + bw[c] * reg_tcn[c])
    return np.where(pmv >= th[c], DEFAULT_VALUES[c], regb)


def defreal(pp, rt, reg_tcn, pdf_tcn, y, D, th=V7["thresholds"],
            cw=V7["clf_tcn_w"], bw=V7["blend_w"]) -> dict:
    """决策口径分解：{c: (frac_dec, frac_truth, acc_dec, acc_real_dec, acc_real_truth)}。

    y = {c: numpy 数组}（含 NaN），D = default_mask(DataFrame)。
    ⚠️ 掩码必须用**模型自己的判默认**（`pm >= th`），不能用 `default_mask`：
      真值掩码下的 acc_def 恒 == 1.0000（默认点上预测恰好等于默认值），零区分度。
      §4.1 表里 0.9881 那组是**全量 80 井部署工件**在 screen 井上的自评（§4.8 判为泄漏），
      不能拿 72 井 harness 的数字去和它对。
    frac_dec > frac_truth 就是「判默认过度」——§4.2 已证明这是本项目唯一的系统性损失模式
    （v6 判默认占比 POR 0.7536 / PERM 0.7781 / SW 0.8626 vs 真值 0.6003 / 0.6063 / 0.7340）。

    本函数已自检：v6 配置下 frac_dec 复现 0.7536/0.7781/0.8626，v7 配置 Total = 73.7491。
    """
    out = {}
    for c in TARGETS:
        yv = np.asarray(y[c], np.float64)
        m = np.isfinite(yv)
        p = preds_at(pp, rt, reg_tcn, pdf_tcn, yv, c, th, cw, bw)
        pm = (1 - cw[c]) * pp[c] + cw[c] * pdf_tcn[c]
        dec = pm >= th[c]
        a = D[c].to_numpy()
        out[c] = (float(dec[m].mean()), float(a[m].mean()),
                  float(acc_vec(c, p[m & dec], yv[m & dec]).mean()),
                  float(acc_vec(c, p[m & ~dec], yv[m & ~dec]).mean()),
                  float(acc_vec(c, p[m & ~a], yv[m & ~a]).mean()))
    return out


def per_well_total(pp, rt, reg_tcn, pdf_tcn, y, m, th, cw, bw, widx, n_well) -> np.ndarray:
    """8 口井各自的 Total 贡献（分）；加总 == 官方 Total（逐点等权）。"""
    out = np.zeros(n_well)
    for c in TARGETS:
        a = acc_vec(c, preds_at(pp, rt, reg_tcn, pdf_tcn, y, c, th, cw, bw), y[c])
        a = np.where(m[c], a, 0.0)
        nv = int(m[c].sum())
        for i in range(n_well):
            ii = widx == i
            k = int(m[c][ii].sum())
            if k:
                out[i] += 100.0 * WEIGHTS[c] * a[ii].sum() / nv
    return out


def sweep_th(p, rt, reg_tcn, pdf_tcn, y, m, c, cw, bw):
    """固定 blend_w / clf_tcn_w，只扫阈值，返回 (最优 acc, 最优 th)。"""
    mm = m[c]
    pmv = ((1 - cw[c]) * p[c] + cw[c] * pdf_tcn[c])[mm]
    regb = np.power(10.0, (1 - bw[c]) * rt[c] + bw[c] * reg_tcn[c])[mm]
    yy = y[c][mm]
    fd = acc_vec(c, np.full(int(mm.sum()), DEFAULT_VALUES[c]), yy)
    fr = acc_vec(c, regb, yy)
    order = np.argsort(pmv, kind="stable")
    pos = np.searchsorted(pmv[order], TH_GRID, side="left")
    Fd = np.concatenate([[0.0], np.cumsum(fd[order])])
    Fr = np.concatenate([[0.0], np.cumsum(fr[order])])
    n = len(pmv)
    acc = (Fr[pos] + (Fd[n] - Fd[pos])) / n
    j = int(np.argmax(acc))
    return float(acc[j]), float(TH_GRID[j])


# ---------------------------------------------------------------------------
# 训练（clf300 分段器 + reg400 真实段回归，带 weight_tol）
# ---------------------------------------------------------------------------
def fit_clf(Xtr, Xva, valid, Dtr, c, params):
    clf = _clf(params)
    clf.fit(Xtr[valid], Dtr.loc[valid, c].astype(int).to_numpy())
    p = clf.predict_proba(Xva)[:, 1].astype(np.float32)
    del clf
    gc.collect()
    return p


def fit_reg(Xtr, Xva, real, Ytr, c):
    """real = 有效且非默认的联合掩码（与 experiments/eval_v5_deployable.py:124 同口径）。

    weight_tol 是部署回归器的一部分（train.py:278-290），不能关。
    """
    reg = _reg(monotonic=POR_MONOTONIC if c == "POR" else None, columns=list(Xtr.columns))
    yt = transform_target(c, Ytr[c].loc[real]).to_numpy().astype(np.float32)
    yraw = Ytr[c].loc[real].to_numpy().astype(np.float64)
    w = np.abs(np.log10(np.maximum(yraw, EPS))) if c == "PERM" else np.abs(yraw)
    w = np.clip(w, np.percentile(w, 5), np.percentile(w, 95))
    w = (w / w.mean()).astype(np.float32)
    reg.fit(Xtr.loc[real], yt, sample_weight=w)
    r = log10ify(c, reg.predict(Xva)).astype(np.float32)
    del reg
    gc.collect()
    return r


def real_mask(ytr_arr: np.ndarray, Dtr, c) -> np.ndarray:
    return (~np.isnan(ytr_arr)) & (~Dtr[c].to_numpy().astype(bool))


def auc_of(yt, ys) -> float:
    return float(roc_auc_score(yt.astype(int), ys))


# ---------------------------------------------------------------------------
# 特征构造
# ---------------------------------------------------------------------------
def _ctx_for_well(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("DEPTH")
    cols: dict[str, np.ndarray] = {
        "rel_depth": g["DEPTH"].to_numpy() - g["DEPTH"].to_numpy()[0]
    }
    for c in CURVE_COLS:
        s = g[c]
        for k in LAGS:
            cols[f"{c}_lag{k}"] = s.shift(k).to_numpy()
        for w in WINS:
            r = s.rolling(w, center=True, min_periods=1)
            cols[f"{c}_wmean{w}"] = r.mean().to_numpy()
            cols[f"{c}_wstd{w}"] = r.std().fillna(0.0).to_numpy()
        cols[f"{c}_diff1"] = s.diff(1).to_numpy()
    return pd.DataFrame(cols, index=g.index)


def build_features(raw13: pd.DataFrame, ids: np.ndarray,
                   override: pd.DataFrame | None) -> pd.DataFrame:
    """raw13 = 已清洗的 13 列（FEATURES 顺序）。override 非空时把 12 条曲线整体替换
    （逐井归一化后），lag/窗口/差分/物理先验在替换后的曲线上重算 —— 表示一致。
    不复用 make_features 是因为它会把 CNL 物理截断到 [0,100]，把 z 分数削平。"""
    X = raw13.copy()
    if override is not None:
        for c in CURVE_COLS:
            X[c] = override[c].to_numpy()
    X["logRT"] = np.log10(X["RT"].clip(lower=EPS))
    X["logRXO"] = np.log10(X["RXO"].clip(lower=EPS))
    X["RT_over_RXO"] = X["RT"] / X["RXO"].clip(lower=EPS)
    X["POR_D"] = (2.65 - X["DEN"]) / 1.65 * 100.0
    X["POR_S"] = (X["AC"] - 55.5) / (189.0 - 55.5) * 100.0
    X["POR_avg"] = (X["POR_D"] + X["CNL"]) / 2.0
    X["POR_ds"] = X["POR_D"] - X["POR_S"]
    parts = []
    for _, g in X.groupby(ids, sort=False):
        parts.append(_ctx_for_well(g))
    return X.join(pd.concat(parts)).astype(np.float32)


def bounds(ids: np.ndarray) -> list[tuple[int, int]]:
    """df 已按井连续排列 → 直接切段，避免反复布尔索引。"""
    cut = np.flatnonzero(ids[1:] != ids[:-1]) + 1
    st = np.concatenate([[0], cut, [ids.size]])
    return [(int(a), int(b)) for a, b in zip(st[:-1], st[1:])]


def per_well_norm(raw13: pd.DataFrame, ids: np.ndarray, kind: str) -> pd.DataFrame:
    """12 条曲线各自的逐井归一化：z=本井 z 分数，mm=本井 min-max。统计量只用本井。"""
    n = raw13.shape[0]
    out = np.full((n, len(CURVE_COLS)), np.nan, dtype=np.float32)
    for a, b in bounds(ids):
        blk = raw13.iloc[a:b]
        for j, c in enumerate(CURVE_COLS):
            s = blk[c].to_numpy().astype(np.float64)
            if kind == "z":
                sd = np.nanstd(s)
                out[a:b, j] = (s - np.nanmean(s)) / (sd if sd > 0 else 1.0)
            else:
                lo, hi = np.nanmin(s), np.nanmax(s)
                rng = hi - lo
                out[a:b, j] = (s - lo) / (rng if rng > 0 else 1.0)
    return pd.DataFrame(out, columns=CURVE_COLS)


def additions(raw13: pd.DataFrame, ids: np.ndarray) -> dict:
    """基座之外新增的候选列。全部按井分组，无跨井统计量。"""
    n = raw13.shape[0]
    res: dict[str, pd.DataFrame] = {"wellz": per_well_norm(raw13, ids, "z")}

    d = np.full((n, 12), np.nan, dtype=np.float32)     # 去趋势：本点 − 本井 w=201 滚动均值
    for a, b in bounds(ids):
        blk = raw13.iloc[a:b]
        for j, c in enumerate(CURVE_COLS):
            s = blk[c]
            d[a:b, j] = (s - s.rolling(201, center=True, min_periods=1).mean()).to_numpy()
    res["det201m"] = pd.DataFrame(d, columns=[f"det201m_{c}" for c in CURVE_COLS])

    # 注：原题设的"5 次多项式低频包络"实测 2.8%（24/864）的（井,曲线）对会让 LAPACK
    # polyfit 抛 LinAlgError（SVD 不收敛），退化后残差 = 原曲线，列退化且噪声大。
    # 低频包络这个方向已由 det201m（w=201 滚动均值）覆盖，故不重复实现多项式版本。

    d = np.full((n, 3), np.nan, dtype=np.float32)      # 本井相对深度 + 两个分箱
    for a, b in bounds(ids):
        dep = raw13["DEPTH"].to_numpy()[a:b].astype(np.float64)
        lo, hi = np.nanmin(dep), np.nanmax(dep)
        rng = hi - lo
        rng = rng if rng > 0 else 1.0
        r01 = (dep - lo) / rng
        d[a:b, 0] = r01
        d[a:b, 1] = np.floor(r01 * 10).clip(0, 9)
        d[a:b, 2] = np.floor(r01 * 20).clip(0, 19)
    res["depth"] = pd.DataFrame(d, columns=["rel_depth01", "dbin10", "dbin20"])
    return res


ADD_COLS: dict[str, list] = {
    "wellz": [f"wellz_{c}" for c in CURVE_COLS],
    "det201m": [f"det201m_{c}" for c in CURVE_COLS],
    "depth": ["rel_depth01", "dbin10", "dbin20"],
}
TAG_TO_ADD = {"plus_wellz": "wellz", "plus_det201m": "det201m", "plus_depth": "depth"}


def load_feats(tag: str | None, base_cols) -> tuple:
    """按变体名取出 (Xtr, Xva)；tag=None → 基座。带正确列名（POR 单调约束要用）。"""
    if tag in (None, "wellz", "wellmm"):
        key = "base" if tag is None else tag
        return (pd.DataFrame(np.load(f"{FEATDIR}/{key}_tr.npy"), columns=base_cols),
                pd.DataFrame(np.load(f"{FEATDIR}/{key}_va.npy"), columns=base_cols))
    k = TAG_TO_ADD[tag]
    bt = np.hstack([np.load(f"{FEATDIR}/base_tr.npy"), np.load(f"{FEATDIR}/add_{k}_tr.npy")])
    bv = np.hstack([np.load(f"{FEATDIR}/base_va.npy"), np.load(f"{FEATDIR}/add_{k}_va.npy")])
    cols = list(base_cols) + ADD_COLS[k]
    assert bt.shape[1] == bv.shape[1] == len(cols), (bt.shape, bv.shape, len(cols))
    return pd.DataFrame(bt, columns=cols), pd.DataFrame(bv, columns=cols)


def main() -> None:
    """--ctx v7|v8 切换基座上下文窗口；--only TAG 只完整验证指定候选（跳过阶段 1 的筛选）。"""
    global REC, LAGS, WINS, FEATDIR, OUT_JSON
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", choices=sorted(CTX_PRESETS), default="v7",
                    help="基座上下文窗口：v7=±10/w21(225)，v8=±60/w41(321)")
    ap.add_argument("--only", default="", help="只完整验证该候选（跳过阶段 1 筛选），如 plus_wellz")
    a = ap.parse_args()
    LAGS, WINS = list(CTX_PRESETS[a.ctx][0]), list(CTX_PRESETS[a.ctx][1])
    F.LAG_OFFSETS, F.WIN_SIZES = list(LAGS), list(WINS)
    FEATDIR = os.path.join(RUNS, f"wellnorm_feat_{a.ctx}")
    OUT_JSON = os.path.join(RUNS, f"wellnorm_{a.ctx}.json")
    os.makedirs(FEATDIR, exist_ok=True)
    say("=" * 104)
    say(f"逐井尺度归一化实验（基座 {a.ctx} = ±{abs(LAGS[0])}/w{max(WINS)}，"
        f"v7 融合配置；sklearn-only）  锚点 {ANCHOR[a.ctx]}")
    say("=" * 104)

    wells = load_wells("./train", has_labels=True)
    ids = [w["logId"].iloc[0] for w in wells]
    va_set = set(ids[N_TRAIN_WELLS:])
    tr = pd.concat([w for w in wells if w["logId"].iloc[0] not in va_set], ignore_index=True)
    va = pd.concat([w for w in wells if w["logId"].iloc[0] in va_set], ignore_index=True)
    del wells
    gc.collect()
    assert len(tr) == BND and len(va) == N_VA, (len(tr), len(va))
    va_ids = list(dict.fromkeys(va["logId"].to_numpy()))
    assert len(va_ids) == 8, va_ids
    ST["va_ids"] = va_ids
    say(f"训练 {len(tr):,} 点 / 验证 {len(va):,} 点（{len(va_ids)} 井）  [{secs():.0f}s]")

    Z = np.load(os.path.join(RUNS, "tcn_all80_parts.npz"))
    reg_tcn = {c: np.float32(Z[f"reg_{c}"][BND:]) for c in TARGETS}
    pdf_tcn = {c: np.float32(Z[f"pdf_{c}"][BND:]) for c in TARGETS}
    del Z

    Ytr, Yva = make_targets(tr), make_targets(va)
    Dtr, Dva = default_mask(Ytr), default_mask(Yva)
    ytr = {c: Ytr[c].to_numpy() for c in TARGETS}
    yva = {c: Yva[c].to_numpy() for c in TARGETS}
    m_va = {c: ~np.isnan(yva[c]) for c in TARGETS}
    widx = (pd.Categorical(va["logId"].to_numpy(), categories=va_ids).codes
            ).astype(np.int64)
    del Yva
    gc.collect()

    # ── 阶段 0a：打分口径自检
    say("\n" + "─" * 104)
    say("阶段 0a：打分口径自检（screen_tree_cache = v7 已部署树的 8 井预测）")
    say("─" * 104)
    cache = joblib.load(os.path.join(RUNS, "screen_tree_cache.joblib"))
    pr, pp = cache["reg"], cache["p_def"]
    pred = {c: preds_at(pp, pr, reg_tcn, pdf_tcn, yva, c, V7["thresholds"],
                        V7["clf_tcn_w"], V7["blend_w"]) for c in TARGETS}
    exact = total_score(pred, yva)
    say(f"缓存锚点 total = {exact:.4f}  vs 部署 {DEPLOY_V7}  (差 {exact-DEPLOY_V7:+.4f})")
    bsc0 = per_well_total(pp, pr, reg_tcn, pdf_tcn, yva, m_va, V7["thresholds"],
                          V7["clf_tcn_w"], V7["blend_w"], widx, 8)
    say(f"  逐井贡献加总 {bsc0.sum():.4f}（应与锚点一致）；逐井 {np.round(bsc0, 2).tolist()}")
    if abs(exact - DEPLOY_V7) > 0.05 or abs(bsc0.sum() - exact) > 1e-6:
        say("口径自检失败 → 终止（打分有偏，后面所有 Δ 不可信）。")
        return

    # ── 特征表：算一次落盘
    say(f"\n基座特征（{a.ctx}）构造 + 落盘 ...")
    Xtr = F.make_features(tr, context=True, priors=True)
    Xva = F.make_features(va, context=True, priors=True)
    NB = Xtr.shape[1]
    assert Xtr.shape[1] == Xva.shape[1] == NB, (Xtr.shape, Xva.shape)
    say(f"  基座列数 = {NB}（{a.ctx}；期望 225 / 321）")
    base_cols = list(Xtr.columns)
    raw_tr = Xtr[FEATURES].copy()          # 已清洗的 13 列（哨兵→NaN，CNL 已截断）
    raw_va = Xva[FEATURES].copy()
    np.save(f"{FEATDIR}/base_tr.npy", Xtr.to_numpy(dtype=np.float32))
    np.save(f"{FEATDIR}/base_va.npy", Xva.to_numpy(dtype=np.float32))
    del Xtr, Xva
    gc.collect()
    say(f"  base 657,489×{NB} / 72,779×{NB} 落盘  [{secs():.0f}s]  峰值 RSS {rss_gb():.2f} GB")

    ids_tr, ids_va = tr["logId"].to_numpy(), va["logId"].to_numpy()
    chk = build_features(raw_tr.iloc[:2000], ids_tr[:2000], None)
    ref = F.make_features(tr.iloc[:2000], context=True, priors=True)
    dmax = float(np.nanmax(np.abs(chk.to_numpy() - ref.to_numpy())))
    say(f"  本地构造器 vs make_features 最大绝对差 = {dmax:.2e}（应 ≈0）")
    del chk, ref
    gc.collect()

    z_tr, z_va = per_well_norm(raw_tr, ids_tr, "z"), per_well_norm(raw_va, ids_va, "z")
    for tag, kind in [("wellz", "z"), ("wellmm", "mm")]:
        src_tr = z_tr if kind == "z" else per_well_norm(raw_tr, ids_tr, "mm")
        src_va = z_va if kind == "z" else per_well_norm(raw_va, ids_va, "mm")
        Xt = build_features(raw_tr, ids_tr, src_tr)
        Xv = build_features(raw_va, ids_va, src_va)
        assert Xt.shape[1] == Xv.shape[1] == NB, (Xt.shape, Xv.shape, NB)
        np.save(f"{FEATDIR}/{tag}_tr.npy", Xt.to_numpy(dtype=np.float32))
        np.save(f"{FEATDIR}/{tag}_va.npy", Xv.to_numpy(dtype=np.float32))
        del Xt, Xv, src_tr, src_va
        gc.collect()
        say(f"  {tag:7s} 替换型 {NB} 特征落盘  [{secs():.0f}s]  峰值 RSS {rss_gb():.2f} GB")
    del z_tr, z_va
    gc.collect()

    add_tr, add_va = additions(raw_tr, ids_tr), additions(raw_va, ids_va)
    for k in ("wellz", "det201m", "depth"):
        np.save(f"{FEATDIR}/add_{k}_tr.npy", add_tr[k].to_numpy(dtype=np.float32))
        np.save(f"{FEATDIR}/add_{k}_va.npy", add_va[k].to_numpy(dtype=np.float32))
    del add_tr, add_va, raw_tr, raw_va, tr, va
    gc.collect()
    say(f"  新增列 wellz 12 / det201m 12 / depth 3 落盘  [{secs():.0f}s]")

    # ── 阶段 0b：重拟合基座自检
    say("\n" + "─" * 104)
    say("阶段 0b：重拟合基座自检（clf300 + reg400 + weight_tol + v7 阈值）")
    say("─" * 104)
    Xtr, Xva = load_feats(None, base_cols)
    bp, br, sauc, sacc = {}, {}, {}, {}
    for c in TARGETS:
        v = ~np.isnan(ytr[c])
        bp[c] = fit_clf(Xtr, Xva, v, Dtr, c, None)
        br[c] = fit_reg(Xtr, Xva, real_mask(ytr[c], Dtr, c), Ytr, c)
        m = m_va[c]
        sauc[c] = auc_of(Dva[c].to_numpy()[m], bp[c][m])
        sacc[c] = float(acc_vec(c, preds_at(bp, br, reg_tcn, pdf_tcn, yva, c,
                                             V7["thresholds"], V7["clf_tcn_w"],
                                             V7["blend_w"])[m], yva[c][m]).mean())
    base_tot = float(sum(WEIGHTS[c] * sacc[c] for c in TARGETS) * 100)
    ST["base_tot"] = base_tot
    bsc = per_well_total(bp, br, reg_tcn, pdf_tcn, yva, m_va, V7["thresholds"],
                         V7["clf_tcn_w"], V7["blend_w"], widx, 8)
    ok = abs(base_tot - ANCHOR[a.ctx]) <= 0.05
    say(f"重拟合基座 total = {base_tot:.3f}  vs 锚点 {ANCHOR[a.ctx]}  "
        f"(差 {base_tot-ANCHOR[a.ctx]:+.3f})  →  {'通过' if ok else '超差：Δ 只能当相对量看'}")
    say(f"  AUC  POR {sauc['POR']:.4f} / PERM {sauc['PERM']:.4f} / SW {sauc['SW']:.4f}")
    say(f"  acc  POR {sacc['POR']:.4f} / PERM {sacc['PERM']:.4f} / SW {sacc['SW']:.4f}"
        f"  [{secs():.0f}s]")
    say(f"  逐井贡献 {np.round(bsc, 2).tolist()}（加总 {bsc.sum():.3f}，应 == total）")
    # 硬门槛只留内部一致性（逐井加总 == total）。锚点偏差是软信号：
    # Δ 一律相对**本次重拟合的基座**算，不依赖锚点；阶段 0a 已用缓存锚点做过硬自检。
    assert abs(bsc.sum() - base_tot) < 1e-6, (bsc.sum(), base_tot)
    ST["base_per_well"] = bsc.tolist()
    # def/real 分解（真值分段，v7 固定阈值）：默认段占比 0.60–0.73 且 acc 已 0.98+，
    # 默认段每丢 1e-2 的 acc 会被 1.5 倍的权重差放大，抵消真实段 1.5e-2 的改善。
    # 所以「不得降低 def acc」是判断候选是否会在测试集翻转的硬筛选条件。
    ST["base_defreal"] = defreal(bp, br, reg_tcn, pdf_tcn, yva, Dva)
    _b = ST["base_defreal"]
    say("  基座决策口径(v7 阈值)：")
    for c in TARGETS:
        fd, ft, ad, ar, art = _b[c]
        say(f"    {c:4s} 判默认 {fd:.4f} / 真值 {ft:.4f}  (偏 {fd-ft:+.4f})  "
            f"acc判默认 {ad:.4f}  acc真实段 {ar:.4f}  (真值真实段 {art:.4f})")
    REC.append({"name": f"BASE{NB} selfcheck(clf300+reg400) ctx={a.ctx}", "n": NB,
                "total": base_tot, "anchor": ANCHOR[a.ctx],
                "delta_vs_deploy": base_tot - ANCHOR[a.ctx],
                "por_acc": sacc["POR"], "perm_acc": sacc["PERM"], "sw_acc": sacc["SW"],
                "por_auc": sauc["POR"], "perm_auc": sauc["PERM"], "sw_auc": sauc["SW"],
                "per_well": bsc.tolist()})
    dump()
    if not ok:
        say("  注意：基座自检超差 —— Δ 仍相对本次重拟合基座算，但绝对水平不可信。")
    del Xtr, Xva
    gc.collect()

    # ── 阶段 1：clf60 快速 AUC 筛选（--only 时跳过，候选已指定）
    fb_auc, fast, passed, top = None, [], [], []
    if not a.only:
        say("\n" + "─" * 104)
        say(f"阶段 1：快速筛选（clf max_iter={SCREEN_CLF['max_iter']}，"
            f"base 与候选同配置 → 只看 ΔAUC）")
        say("─" * 104)
        Xtr, Xva = load_feats(None, base_cols)
        fb = {c: fit_clf(Xtr, Xva, ~np.isnan(ytr[c]), Dtr, c, SCREEN_CLF) for c in TARGETS}
        fb_auc = {c: auc_of(Dva[c].to_numpy()[m_va[c]], fb[c][m_va[c]]) for c in TARGETS}
        say(f"  base(fast) POR {fb_auc['POR']:.4f} / PERM {fb_auc['PERM']:.4f} "
            f"/ SW {fb_auc['SW']:.4f}  [{secs():.0f}s]")
        for c in TARGETS:
            del fb[c]
        gc.collect()

        for tag, _kind, _a in CANDS:
            Xt, Xv = load_feats(tag, base_cols)
            row = {"name": tag, "n": int(Xt.shape[1])}
            for c in TARGETS:
                au = auc_of(Dva[c].to_numpy()[m_va[c]],
                            fit_clf(Xt, Xv, ~np.isnan(ytr[c]), Dtr, c, SCREEN_CLF)[m_va[c]])
                row[f"{c.lower()}_auc"] = au
                row[f"{c.lower()}_dauc"] = au - fb_auc[c]
                say(f"  {tag:13s} {c:5s} AUC {au:.4f} (Δ {au-fb_auc[c]:+.4f})  [{secs():.0f}s]")
            row["max_dauc"] = max(row[f"{c.lower()}_dauc"] for c in TARGETS)
            row["sum_dauc"] = sum(row[f"{c.lower()}_dauc"] for c in TARGETS)
            fast.append(row)
            REC.append(row)
            dump()
            del Xt, Xv
            gc.collect()
        del Xtr, Xva
        gc.collect()
        say(f"\n阶段 1 结束；峰值 RSS {rss_gb():.2f} GB  [{secs():.0f}s]")
        passed = [r["name"] for r in fast if r["max_dauc"] >= GATE]
        top = [r["name"] for r in sorted(fast, key=lambda r: (r["sum_dauc"], r["max_dauc"]),
                                         reverse=True)[:2]]
        todo = list(dict.fromkeys(passed + top))
        left = 1290.0 - secs()
        afford = max(1, int(left // 190))
        if len(todo) > afford:
            say(f"  时间预算（剩 {left:.0f}s，单候选完整口径约 185s）→ 只验证前 {afford} 个")
            todo = todo[:afford]
    else:
        todo = [a.only]
        say(f"\n--only {a.only} → 跳过阶段 1 筛选，直接完整验证")
    say("\n" + "─" * 104)
    say(f"阶段 2：完整口径验证（clf300 + reg400 + weight_tol + 阈值扫 + 逐井 + 留一井 jackknife）")
    say(f"  ΔAUC≥{GATE} 通过：{passed or '（无）'}；sum(ΔAUC) 前二保底：{top}")
    say(f"  实际验证：{todo}")
    say("─" * 104)

    for tag in todo:
        Xt, Xv = load_feats(tag, base_cols)
        cth, cacc, cauc, cp, cr = {}, {}, {}, {}, {}
        for c in TARGETS:
            v = ~np.isnan(ytr[c])
            cp[c] = fit_clf(Xt, Xv, v, Dtr, c, None)
            cr[c] = fit_reg(Xt, Xv, real_mask(ytr[c], Dtr, c), Ytr, c)
            cauc[c] = auc_of(Dva[c].to_numpy()[m_va[c]], cp[c][m_va[c]])
            cacc[c] = float(acc_vec(c, preds_at(cp, cr, reg_tcn, pdf_tcn, yva, c,
                                                V7["thresholds"], V7["clf_tcn_w"],
                                                V7["blend_w"])[m_va[c]], yva[c][m_va[c]]).mean())
            sopt, th = sweep_th(cp, cr, reg_tcn, pdf_tcn, yva, m_va, c,
                                V7["clf_tcn_w"], V7["blend_w"])
            cth[c] = th
            say(f"  {tag:13s} {c:5s} AUC {cauc[c]:.4f} (Δvs基座 {cauc[c]-sauc[c]:+.4f}) "
                f"acc@v7th {cacc[c]:.4f} acc@opt {sopt:.4f} th {th:.3f}(v7 {V7['thresholds'][c]:.3f})"
                f"  [{secs():.0f}s]")
        csc = per_well_total(cp, cr, reg_tcn, pdf_tcn, yva, m_va, cth,
                             V7["clf_tcn_w"], V7["blend_w"], widx, 8)
        dw = csc - bsc
        d = float(dw.sum())
        lo, hi = d - float(dw.max()), d - float(dw.min())
        v7_tot = float(sum(WEIGHTS[c] * cacc[c] for c in TARGETS) * 100)
        pos = int((dw > 0).sum())
        cdr = defreal(cp, cr, reg_tcn, pdf_tcn, yva, Dva)
        rec = {"name": f"{tag} [full clf300+reg400]", "n": int(Xt.shape[1]),
               "ctx": a.ctx,
               "total_opt": float(csc.sum()), "total_v7th": v7_tot,
               "delta_opt": d, "delta_v7th": v7_tot - base_tot,
               "por_acc": cacc["POR"], "perm_acc": cacc["PERM"], "sw_acc": cacc["SW"],
               "por_auc": cauc["POR"], "perm_auc": cauc["PERM"], "sw_auc": cauc["SW"],
               "th_opt": cth, "loo": [lo, hi], "loo_mean": d * 7 / 8,
               "pos_wells": pos, "n_wells": 8,
               # 决策口径分解 [判默认占比, 真值占比, acc判默认, acc真实段(判真实), acc真实段(真值真实段)]
               "defreal": {c: cdr[c] for c in TARGETS},
               "defreal_delta": {c: [cdr[c][i] - ST["base_defreal"][c][i]
                                     for i in range(5)] for c in TARGETS},
               "per_well_delta": [round(float(x), 4) for x in dw],
               "patho_idx": va_ids.index(PATHO),
               "patho_delta": float(dw[va_ids.index(PATHO)])}
        REC.append(rec)
        dump()
        say(f"  → Total@最优阈值 {csc.sum():.3f} (Δ {d:+.3f}) | Total@v7阈值 {v7_tot:.3f} "
            f"(Δ {v7_tot-base_tot:+.3f})")
        say(f"    逐井 Δ {np.round(dw, 3).tolist()}  正号 {pos}/8   "
            f"留一井 jackknife [{lo:+.3f}, {hi:+.3f}] {'跨 0' if lo <= 0 <= hi else '不跨 0'}")
        say(f"    病态井 {PATHO[:8]}… Δ = {rec['patho_delta']:+.3f}")
        for c in TARGETS:
            del cp[c]
        gc.collect()
        del Xt, Xv
        gc.collect()

    # ── 汇总
    say("\n" + "=" * 104)
    say("汇总（screen 单切分，官方口径；基座 %s 锚点 %.3f，本次重拟合基座 = %.3f）"
        % (a.ctx, ANCHOR[a.ctx], base_tot))
    say("=" * 104)
    if fb_auc is not None:
        say(f"{'候选':24s}{'列':>5s}{'AUC POR':>10s}{'PERM':>9s}{'SW':>9s}{'sumΔAUC':>10s}")
        say(f"{'base(fast clf60)':24s}{NB:5d}{fb_auc['POR']:10.4f}{fb_auc['PERM']:9.4f}"
            f"{fb_auc['SW']:9.4f}{0.0:+10.4f}")
        for r in fast:
            say(f"{r['name']:24s}{r['n']:5d}{r['por_auc']:10.4f}{r['perm_auc']:9.4f}"
                f"{r['sw_auc']:9.4f}{r['sum_dauc']:+10.4f}")
    say(f"\n{'候选':26s}{'列':>5s}{'AUC POR':>10s}{'PERM':>9s}{'SW':>9s}"
        f"{'Total@opt':>10s}{'Δ':>9s}{'Δ@v7th':>9s}{'正井':>6s}{'留一井 jackknife':>24s}")
    for r in REC:
        if "loo" not in r:
            continue
        jk = f"[{r['loo'][0]:+.3f},{r['loo'][1]:+.3f}]"
        say(f"{r['name'][:26]:26s}{r['n']:5d}{r['por_auc']:10.4f}{r['perm_auc']:9.4f}"
            f"{r['sw_auc']:9.4f}{r['total_opt']:10.3f}{r['delta_opt']:+9.3f}"
            f"{r['delta_v7th']:+9.3f}{str(r['pos_wells'])+'/8':>6s}{jk:>24s}")
    say(f"\n峰值 RSS {rss_gb():.2f} GB（本机 16 GB）；结果 {OUT_JSON}；总耗时 {secs():.0f}s")


if __name__ == "__main__":
    main()
