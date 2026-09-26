"""上下文窗口宽度实验：官方口径下重测「更宽的 lag / 窗口」是否提高分段器 AUC。

为什么这是唯一还没试过的杠杆（2026-09-26）：
  `src/feature.py` 的 `LAG_OFFSETS = [-10,-5,-3,-2,-1,1,2,3,5,10]`、`WIN_SIZES = [5,11,21]`
  是自项目始就写死的常量，从未调过。而 POR 的 lag-1 自相关是 **0.967**，
  相关长度 1/ln(0.967) ≈ 30 个点（3 m）；当前最远只取到 ±10，
  0.967^10 = 0.718 —— 最远一阶滞后还保留 72% 的相关性，等于把大半时间结构丢在窗外。

  已证伪的家族（`diag_loss_budget.py` + `tune_decision_struct.py`）：
    回归平滑 +0.000 / 决策中值滤波 −1.029 / Viterbi +0.006 / 三目标共识 全负 /
    分类器重训 +0.088 但属联合过拟合。全部 ≤ 噪声带 ±0.3。
  这些共同指向：AUC 0.897 是**当前 225 个特征的信息上限**。
  唯一没试的是「给模型更多特征」——同一批 13 条曲线，但把时间窗口拉宽。

可行性：`predict.py:60` 里 TCN 走 `context=False`，树与 TCN 的特征通路**完全独立**，
所以只改树的 `LAG_OFFSETS` / `WIN_SIZES` 不影响 TCN，模型仍可原样融合部署。
`POR_MONOTONIC = {"DEN": -1, "AC": 1}` 按列名匹配、缺省为 0，加列不会破坏单调约束。

评估口径与部署一致：树回归器带 weight_tol 重训（`train.py:278-290`），
按 v7 的 blend_w / clf_tcn_w 与缓存的 TCN 前向融合，阈值在验证集上扫。

用法: python experiments/tune_ctx_width.py
"""
from __future__ import annotations
import gc, os, sys, time, warnings
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

RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
N_TRAIN_WELLS = 72
V7 = {"blend_w": {"POR": 0.50, "PERM": 0.50, "SW": 0.75},
      "thresholds": {"POR": 0.13, "PERM": 0.27, "SW": 0.05},
      "clf_tcn_w": {"POR": 0.60, "PERM": 0.70, "SW": 0.30}}
TH_GRID = np.arange(0.01, 0.991, 0.01)
T0 = time.time()

# 候选配置：(名称, LAG_OFFSETS, WIN_SIZES)
CFG = [
    ("cur ±10/w21",  [-10, -5, -3, -2, -1, 1, 2, 3, 5, 10], [5, 11, 21]),
    ("±30/w21",      [-30, -20, -10, -5, -3, -2, -1, 1, 2, 3, 5, 10, 20, 30], [5, 11, 21]),
    ("±30/w41",      [-30, -20, -10, -5, -3, -2, -1, 1, 2, 3, 5, 10, 20, 30], [5, 11, 21, 41]),
    ("±60/w41",      [-60, -30, -20, -10, -5, -3, -2, -1, 1, 2, 3, 5, 10, 20, 30, 60],
                     [5, 11, 21, 41]),
]


def acc_vec(c, pred, true):
    p = np.asarray(pred, np.float64); t = np.asarray(true, np.float64)
    if c == "PERM":
        return np.maximum(0.0, 1.0 - np.abs(np.log10(np.maximum(p / np.maximum(t, EPS), EPS))))
    tol = 0.08 if c == "POR" else 0.05
    return np.maximum(0.0, 1.0 - np.abs(p - t) / (tol * (np.abs(t) + EPS)))


def log10ify(c, r):
    """回归输出统一折到 log10：PERM 回归器直接输出 log10，POR/SW 输出物理量。"""
    r = np.asarray(r, np.float64)
    if c == "PERM":
        return r
    return np.log10(np.maximum(r, EPS))


def acc_at_th(pm_tree, c, rt_log, reg_tcn, pdf_tcn, y, m, th, cw, bw):
    """cw 只用于**分段概率**融合（树 p_def + TCN pdf）；bw 只用于**回归**融合。

    两者都不能用 TCN 的回归输出 —— 见 2026-09-26 踩坑：把 reg 当 pdf 会让
    阈值扫不到 v7 工作点（acc 恒为 0.6009、最优 th 退到网格下界 0.01）。
    """
    pmv = (1 - cw) * pm_tree + cw * pdf_tcn
    regb = np.power(10.0, (1 - bw) * rt_log + bw * reg_tcn)
    pred = np.where(pmv >= th, DEFAULT_VALUES[c], regb)
    return float(acc_vec(c, pred[m], y[m]).mean())


def sweep_acc(pm_tree, c, rt_log, reg_tcn, pdf_tcn, y, m, cw, bw):
    """固定 blend_w / clf_tcn_w，只在阈值上扫，返回最优 (acc, th)。

    pm_tree / rt_log 必须是**验证集全量**（长度 == len(m)），掩码由调用方传入。
    """
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


def main() -> None:
    wells = load_wells("./train", has_labels=True)
    ids = [w["logId"].iloc[0] for w in wells]
    va_ids = set(ids[N_TRAIN_WELLS:])
    tr = pd.concat([w for w in wells if w["logId"].iloc[0] not in va_ids], ignore_index=True)
    va = pd.concat([w for w in wells if w["logId"].iloc[0] in va_ids], ignore_index=True)
    Ytr, Yva = make_targets(tr), make_targets(va)
    Dtr, Dva = default_mask(Ytr), default_mask(Yva)
    ytr = {c: Ytr[c].to_numpy() for c in TARGETS}
    yva = {c: Yva[c].to_numpy() for c in TARGETS}

    Z = np.load(os.path.join(RUNS, "tcn_all80_parts.npz"))
    bnd = len(tr)
    ens = {c: np.float32(Z[f"reg_{c}"][bnd:]) for c in TARGETS}
    pdf = {c: np.float32(Z[f"pdf_{c}"][bnd:]) for c in TARGETS}

    print("=" * 100)
    print("上下文窗口宽度实验（树与 TCN 特征通路独立；blend_w/clf_tcn_w 固定 v7，阈值在验证集扫）")
    print("=" * 100)
    print(f"训练 {bnd:,} 点 / 验证 {len(va):,} 点   [{time.time()-T0:.0f}s]")

    summary = []
    for name, lags, wins in CFG:
        F.LAG_OFFSETS = list(lags)
        F.WIN_SIZES = list(wins)
        Xtr = F.make_features(tr, context=True, priors=True)
        Xva = F.make_features(va, context=True, priors=True)
        ncol = Xtr.shape[1]

        tot = 0.0
        print(f"\n{'─'*100}")
        print(f"配置 {name}: {ncol} 特征   [{time.time()-T0:.0f}s]")
        print(f"{'目标':5s}{'AUC':>8s}{'acc@v7th':>10s}{'acc@最优th':>12s}{'最优th':>8s}"
              f"{'acc@v7贡献':>12s}")

        for c in TARGETS:
            v = ~np.isnan(ytr[c])
            clf = _clf()
            clf.fit(Xtr.loc[v], Dtr.loc[v, c].astype(int).to_numpy())
            p_va = clf.predict_proba(Xva)[:, 1].astype(np.float32)
            del clf
            gc.collect()
            m0 = ~np.isnan(yva[c])
            auc = float(roc_auc_score(Dva[c].to_numpy().astype(int)[m0], p_va[m0]))

            # 回归器：weight_tol 加权（与 train.py:278-290 部署口径同源）
            real = v & (~Dtr[c].to_numpy().astype(bool))
            reg = _reg(monotonic=POR_MONOTONIC if c == "POR" else None, columns=list(Xtr.columns))
            yt = transform_target(c, Ytr[c].loc[real]).to_numpy()
            yraw = Ytr[c].loc[real].to_numpy().astype(np.float64)
            w = np.abs(np.log10(np.maximum(yraw, EPS))) if c == "PERM" else np.abs(yraw)
            w = np.clip(w, np.percentile(w, 5), np.percentile(w, 95))
            w = (w / w.mean()).astype(np.float32)
            reg.fit(Xtr.loc[real], yt, sample_weight=w)
            rt_va = log10ify(c, reg.predict(Xva)).astype(np.float32)
            del reg
            gc.collect()

            m = ~np.isnan(yva[c])
            assert len(p_va) == len(rt_va) == len(ens[c]) == len(pdf[c]) == len(m), (
                len(p_va), len(rt_va), len(m), len(ens[c]), len(pdf[c]))
            a7 = acc_at_th(p_va, c, rt_va, ens[c], pdf[c], yva[c], m,
                           V7["thresholds"][c], V7["clf_tcn_w"][c], V7["blend_w"][c])
            ab, thb = sweep_acc(p_va, c, rt_va, ens[c], pdf[c], yva[c], m,
                                V7["clf_tcn_w"][c], V7["blend_w"][c])
            tot += WEIGHTS[c] * ab * 100
            print(f"{c:5s}{auc:8.4f}{a7:10.4f}{ab:12.4f}{thb:8.3f}"
                  f"{WEIGHTS[c]*ab*100:12.3f}")
            summary.append({"cfg": name, "target": c, "auc": auc, "acc_best": ab,
                            "th_best": thb, "total": None})
        print(f"  Total = {tot:.3f}   [{time.time()-T0:.0f}s]")
        summary.append({"cfg": name, "target": "TOTAL", "auc": None,
                        "acc_best": None, "th_best": None, "total": tot})
        del Xtr, Xva
        gc.collect()

    print("\n" + "=" * 100)
    print("汇总：AUC 与 Total（部署 v7 = 73.749）")
    print("=" * 100)
    tot_by_cfg = {r["cfg"]: r["total"] for r in summary if r["total"] is not None}
    print(f"{'配置':14s}{'AUC POR':>9s}{'PERM':>7s}{'SW':>7s}{'Total':>9s}{'Δ':>9s}")
    base_t = None
    for name, _, _ in CFG:
        row = {r["target"]: r["auc"] for r in summary if r["cfg"] == name and r["auc"] is not None}
        t = tot_by_cfg[name]
        if base_t is None:
            base_t = t
        print(f"{name:14s}{row['POR']:9.4f}{row['PERM']:7.4f}{row['SW']:7.4f}"
              f"{t:9.3f}{t-base_t:+9.3f}")
    best = max(tot_by_cfg.items(), key=lambda kv: kv[1])
    print(f"\n  最优 = {best[0]}  {best[1]:.3f}  (Δ {best[1]-base_t:+.3f})")
    print(f"  screen 噪声 ≈ ±0.3，晋级线 ≥ +0.05；|Δ| = {abs(best[1]-base_t):.3f} "
          f"→ {'超出噪声带' if abs(best[1]-base_t) >= 0.3 else '在噪声带内'}")
    print(f"  耗时 {time.time()-T0:.1f}s")


if __name__ == "__main__":
    main()
