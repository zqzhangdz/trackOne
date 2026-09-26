"""决策层结构实验：官方口径下重测「推理后平滑 / 中值滤波 / Viterbi / 三目标共识」。

为什么这次值得重测（历史缺口）：
  EXPERIMENTS.md 里 推理后平滑 / 段内 β 收缩 / Viterbi / 中值滤波分段 / 三目标共识决策
  全部被否决，但当时的 PERM 口径比官方严 3～20 倍（见 CLAUDE.md §11，2026-09-26 核验）。
  EXPERIMENTS.md:42 明确写着「M2 结构化分段在当前回归质量下无效 … M2 推迟到 M3 之后复验」，
  而 M3 之后这个复验**从未执行**。那是本项目唯一一条真正未验证的路线。

关键区分（`diag_loss_budget.py` 已证明的另一半）：
  回归输出上的平滑是**已证伪**的（真值分段下 段内中值 +0.000、中值 k15 −0.483），
  因为真实段的预测已经指标饱和（real_acc 0.19～0.51）。
  但**二值分段决策**上的平滑没有试过。分段器丢掉 +2.587 分，全部预算都在这一步；
  误判高度单向（真真实→判默认 11,250/8,964/7,643 vs 真默认→判真 583/1,219/464），
  沿深度做中值滤波有可能把孤立的误判点翻回来。

做法：72 井训练 / 8 井验证，回归器、blend_w、clf_tcn_w、各目标阈值**全部固定为 v7**
（v7 阈值已是 screen 最优，见本仓库 `tune_official_perm.py`），只改最后那一步二值决策。
纯缓存，不拟合任何模型，不 import torch。

用法: python experiments/tune_decision_struct.py
"""
from __future__ import annotations
import os, sys, time, json
import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.feature import default_mask, load_wells, make_targets
from src.utils import DEFAULT_VALUES, EPS, TARGETS, WEIGHTS

RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
BND = 657_489
V7 = {"blend_w": {"POR": 0.50, "PERM": 0.50, "SW": 0.75},
      "thresholds": {"POR": 0.13, "PERM": 0.27, "SW": 0.05},
      "clf_tcn_w": {"POR": 0.60, "PERM": 0.70, "SW": 0.30}}
T0 = time.time()


def acc_vec(c, pred, true):
    p = np.asarray(pred, np.float64); t = np.asarray(true, np.float64)
    if c == "PERM":
        return np.maximum(0.0, 1.0 - np.abs(np.log10(np.maximum(p / np.maximum(t, EPS), EPS))))
    tol = 0.08 if c == "POR" else 0.05
    return np.maximum(0.0, 1.0 - np.abs(p - t) / (tol * (np.abs(t) + EPS)))


def maj_filter(dec, k):
    """二值决策沿深度的居中大数滤波（k 为奇数）。dec: 1=真实段, 0=默认段。"""
    if k <= 1:
        return dec.copy()
    half = k // 2
    pad = np.pad(dec.astype(np.int64), half, mode="edge")
    cs = np.concatenate([[0], np.cumsum(pad)])          # 滚动和 → 向量化的居中大数
    s = cs[k:] - cs[:-k]
    return (s * 2 > k).astype(np.int64)


def viterbi(pm_dec, th, lam, valid_idx):
    """一阶马尔可夫分段：state 1 = 判真实段（低 pm_dec），state 0 = 默认段（高 pm_dec）。

    pm_dec 高 → 倾向于默认段。发射对数比取「相对阈值 th」的 log-odds：
      r_i = logit(pm_i) − logit(th)
    这样 lam = 0 时逐点 argmax 恰好等于「pm >= th → 默认」，即 v7 的原决策；
    lam > 0 才引入切换代价（越大越倾向长段）。若省略 −logit(th)，lam=0 会退化到
    「pm >= 0.5 才判默认」，与本项目的阈值（0.05～0.27）差一个数量级。
    只在有效行上跑 Viterbi；无效行沿用邻点结果。
    """
    x = pm_dec[valid_idx]
    lg = lambda v: np.log(np.clip(v, 1e-6, 1 - 1e-6) / np.clip(1 - v, 1e-6, 1 - 1e-6))
    r = (lg(x) - lg(th)).astype(np.float64)  # >0 → 默认
    n = len(r)
    if n == 0:
        return np.empty(0, np.int64)

    # 两状态 Viterbi 只需跟踪 d[i] = dp[真实] − dp[默认]，可写成纯标量递推：
    #   d[i] = max(d[i-1], −lam) − max(0, d[i-1] − lam) − 2·r[i]
    # （由 max(dp1, dp0−lam) − max(dp0, dp1−lam) = max(d, −lam) − max(0, d−lam) 推出）
    # lam = 0 时 d[i] = max(d[i-1], 0) − max(0, d[i-1]) − 2r[i] = −2r[i]，
    # 即 path 只看当前点 → 恰好退化为「pm >= th → 默认」的 v7 原决策。
    d = np.empty(n, np.float64)
    d[0] = -2.0 * float(r[0])
    for i in range(1, n):
        p = d[i - 1]
        d[i] = (p if p > -lam else -lam) - (p - lam if p > lam else 0.0) - 2.0 * float(r[i])

    # 回溯：转移只依赖 d[i-1] 与 lam
    path = np.empty(n, np.int64)
    path[-1] = 1 if d[-1] > 0 else 0
    for i in range(n - 1, 0, -1):
        s = int(path[i]); p = d[i - 1]
        if s == 0:
            path[i - 1] = 0 if p <= lam else 1
        else:
            path[i - 1] = 1 if p >= -lam else 0

    out = np.full(len(pm_dec), -1, np.int64)
    out[valid_idx] = path       # state 1 = 真实段，state 0 = 默认段
    return out


def smooth_invalid(out, valid_idx):
    """无效行（无真值）沿用最近的有效行决策。"""
    idx = np.flatnonzero(valid_idx)
    if len(idx) == 0:
        return out
    # np.interp 在前驱有效行之间做常数插值，等价于「沿用最近的有效行」
    out = np.interp(np.arange(len(out)), idx, out[idx]).astype(np.int64)
    return out


def main() -> None:
    st = joblib.load(os.path.join(RUNS, "screen_tree_cache.joblib"))
    Z = np.load(os.path.join(RUNS, "tcn_all80_parts.npz"))

    df = pd.concat(load_wells("./train", has_labels=True), ignore_index=True)
    Y = make_targets(df); DM = default_mask(Y)
    y = {c: Y[c].to_numpy() for c in TARGETS}
    dm = {c: DM[c].to_numpy().astype(bool) for c in TARGETS}

    pm, regb, dec = {}, {}, {}
    for c in TARGETS:
        pm[c] = np.float64((1 - V7["clf_tcn_w"][c]) * st["p_def"][c]
                           + V7["clf_tcn_w"][c] * np.float32(Z[f"pdf_{c}"][BND:]))
        regb[c] = np.power(10.0, (1 - V7["blend_w"][c]) * np.asarray(st["reg"][c], np.float64)
                           + V7["blend_w"][c] * np.float32(Z[f"reg_{c}"][BND:]))
        dec[c] = (pm[c] < V7["thresholds"][c]).astype(np.int64)   # 1 = 真实段

    well_ids = df["logId"].to_numpy()[BND:]
    lids = list(pd.unique(well_ids))

    def total(dec_dict):
        """逐点等权总分，并返回逐井 Δ（相对 baseline）。"""
        accs = {}
        for c in TARGETS:
            pred = np.where(dec_dict[c] == 1, regb[c], DEFAULT_VALUES[c])
            m = ~np.isnan(y[c][BND:])
            accs[c] = float(acc_vec(c, pred[m], y[c][BND:][m]).mean())
        return sum(WEIGHTS[c] * accs[c] * 100 for c in TARGETS), accs

    def per_well(dec_dict):
        d = {}
        for lid in lids:
            wk = np.flatnonzero(well_ids == lid)
            t = 0.0
            for c in TARGETS:
                pred = np.where(dec_dict[c][wk] == 1, regb[c][wk], DEFAULT_VALUES[c])
                m = ~np.isnan(y[c][BND:][wk])
                t += WEIGHTS[c] * acc_vec(c, pred[m], y[c][BND:][wk][m]).mean() * 100
            d[lid] = t
        return d

    base_t, base_acc = total(dec)
    base_wd = per_well(dec)
    print("=" * 100)
    print("决策层结构实验（回归器 / blend_w / clf_tcn_w / 各目标阈值均固定为 v7）")
    print("=" * 100)
    print(f"{'目标':5s}{'真值默认占比':>13s}{'v7 判默认':>11s}{'v7 acc':>9s}{'贡献':>8s}")
    for c in TARGETS:
        m = ~np.isnan(y[c][BND:])
        print(f"{c:5s}{(m & dm[c][BND:]).sum() / m.sum():13.4f}{(dec[c] == 0).mean():11.4f}"
              f"{base_acc[c]:9.4f}{WEIGHTS[c] * base_acc[c] * 100:8.3f}")
    print(f"\n部署 v7 = {base_t:.3f}")
    print()

    results = []

    # ── 1. 二值决策居中大数滤波
    print("─" * 100)
    print("1) 二值决策居中大数滤波（k = 沿深度窗口，奇数；边缘用端点值填充）")
    print("─" * 100)
    for k in (3, 5, 7, 15, 25, 45):
        d = {c: maj_filter(dec[c], k) for c in TARGETS}
        t, a = total(d)
        wd = per_well(d)
        pos = sum(1 for lid in lids if wd[lid] - base_wd[lid] > 0)
        print(f"  k={k:3d}  POR {a['POR']:.4f}  PERM {a['PERM']:.4f}  SW {a['SW']:.4f}   "
              f"Total {t:.3f}  Δ {t-base_t:+.3f}   正号井 {pos}/{len(lids)}")
        results.append({"variant": f"maj_k{k}", "total": t, "delta": t - base_t,
                        "pos_wells": pos, "acc": a})

    # ── 2. Viterbi 分段（切换代价 lam）
    print("\n" + "─" * 100)
    print("2) Viterbi / 一阶马尔可夫分段（lam = 切换代价 log-odds；只在有真值行上运行）")
    print("─" * 100)
    for lam in (0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0):
        d = {}
        for c in TARGETS:
            m = ~np.isnan(y[c][BND:])
            vi = viterbi(pm[c], V7["thresholds"][c], lam, m)
            vi = smooth_invalid(vi, m)
            d[c] = vi
        t, a = total(d)
        wd = per_well(d)
        pos = sum(1 for lid in lids if wd[lid] - base_wd[lid] > 0)
        print(f"  lam={lam:5.2f}  POR {a['POR']:.4f}  PERM {a['PERM']:.4f}  SW {a['SW']:.4f}   "
              f"Total {t:.3f}  Δ {t-base_t:+.3f}   正号井 {pos}/{len(lids)}")
        results.append({"variant": f"viterbi_lam{lam}", "total": t, "delta": t - base_t,
                        "pos_wells": pos, "acc": a})

    # ── 3. 三目标共识决策
    print("\n" + "─" * 100)
    print("3) 三目标共识决策（各目标 pm 直接组合后与本目标阈值比较；推理期三路概率均可用）")
    print("─" * 100)
    pmdef = {c: pm[c] for c in TARGETS}
    cons = {}
    cons["各目标独立(v7)"] = {c: (pmdef[c] >= V7["thresholds"][c]) for c in TARGETS}
    cons["三目标 max"] = {c: (np.maximum.reduce([pmdef[o] for o in TARGETS])
                              >= V7["thresholds"][c]) for c in TARGETS}
    cons["三目标 min"] = {c: (np.minimum.reduce([pmdef[o] for o in TARGETS])
                              >= V7["thresholds"][c]) for c in TARGETS}
    cons["三目标 中位"] = {c: (np.median([pmdef[o] for o in TARGETS])
                               >= V7["thresholds"][c]) for c in TARGETS}
    cons["2/3 多数"] = {c: (np.sum([pmdef[o] >= V7["thresholds"][o] for o in TARGETS], axis=0)
                            >= 2) for c in TARGETS}
    cons["3/3 一致"] = {c: (np.sum([pmdef[o] >= V7["thresholds"][o] for o in TARGETS], axis=0)
                            == 3) for c in TARGETS}
    print(f"  {'方案':16s}{'POR':>8s}{'PERM':>9s}{'SW':>8s}{'Total':>9s}{'Δ':>9s}  "
          f"{'判默认 POR':>11s}{'PERM':>7s}{'SW':>6s}")
    for name, dd in cons.items():
        t, a = total({c: 1 - dd[c].astype(np.int64) for c in TARGETS})
        print(f"  {name:16s}{a['POR']:8.4f}{a['PERM']:9.4f}{a['SW']:8.4f}{t:9.3f}{t-base_t:+9.3f}  "
              f"{dd['POR'].mean():11.4f}{dd['PERM'].mean():7.4f}{dd['SW'].mean():6.4f}")
        results.append({"variant": f"consensus:{name}", "total": t, "delta": t - base_t,
                        "acc": a})

    # ── 4. 共识 + 中值滤波叠加
    print("\n" + "─" * 100)
    print("4) 三目标 2/3 多数 + 中值滤波叠加")
    print("─" * 100)
    for k in (1, 5, 15, 25):
        d = {}
        for c in TARGETS:
            dd = np.sum([pmdef[o] >= V7["thresholds"][o] for o in TARGETS], axis=0) >= 2
            d[c] = maj_filter((~dd).astype(np.int64), k)
        t, a = total(d)
        print(f"  k={k:3d}  POR {a['POR']:.4f}  PERM {a['PERM']:.4f}  SW {a['SW']:.4f}   "
              f"Total {t:.3f}  Δ {t-base_t:+.3f}")
        results.append({"variant": f"cons+maj_k{k}", "total": t, "delta": t - base_t, "acc": a})

    # ── 5. 尺度归一化后的共识（上面「中位」全部判默认的根因：三路 pm 尺度差一个数量级）
    print("\n" + "─" * 100)
    print("5) 尺度归一化共识：pn_c = pm_c / th_c（只用阈值，推理期可用），再池化后与 1.0 比较")
    print("─" * 100)
    pn = {c: pmdef[c] / V7["thresholds"][c] for c in TARGETS}
    sc = {}
    sc["pm/th 中位"] = {c: (np.median([pn[o] for o in TARGETS]) >= 1.0) for c in TARGETS}
    sc["pm/th max"] = {c: (np.maximum.reduce([pn[o] for o in TARGETS]) >= 1.0) for c in TARGETS}
    sc["pm/th min"] = {c: (np.minimum.reduce([pn[o] for o in TARGETS]) >= 1.0) for c in TARGETS}
    sc["pm/th 均值"] = {c: (np.mean([pn[o] for o in TARGETS], axis=0) >= 1.0) for c in TARGETS}
    sc["pm/th 几何均值"] = {c: (np.exp(np.mean([np.log(np.clip(pn[o], 1e-9, None)) for o in TARGETS],
                                    axis=0)) >= 1.0) for c in TARGETS}
    print(f"  {'方案':16s}{'POR':>8s}{'PERM':>9s}{'SW':>8s}{'Total':>9s}{'Δ':>9s}  "
          f"{'判默认 POR':>11s}{'PERM':>7s}{'SW':>6s}")
    for name, dd in sc.items():
        t, a = total({c: 1 - dd[c].astype(np.int64) for c in TARGETS})
        print(f"  {name:16s}{a['POR']:8.4f}{a['PERM']:9.4f}{a['SW']:8.4f}{t:9.3f}{t-base_t:+9.3f}  "
              f"{dd['POR'].mean():11.4f}{dd['PERM'].mean():7.4f}{dd['SW'].mean():6.4f}")
        results.append({"variant": f"scalecons:{name}", "total": t, "delta": t - base_t,
                        "acc": a})

    # ── 汇总
    print("\n" + "=" * 100)
    print("汇总：按 Δ 排序的前 8 名（部署 v7 = %.3f）" % base_t)
    print("=" * 100)
    top = sorted(results, key=lambda r: -r["delta"])[:8]
    for r in top:
        print(f"  {r['variant']:24s} {r['total']:.3f}  Δ {r['delta']:+.3f}")
    best = top[0]
    print(f"\n  最好候选 = {best['variant']}  Δ = {best['delta']:+.3f}")
    print(f"  screen 噪声 ≈ ±0.3，晋级线 ≥ +0.05；本次最好 |Δ| = {abs(best['delta']):.3f} "
          f"→ {'超出噪声带' if abs(best['delta']) >= 0.3 else '在噪声带内'}")
    out = os.path.join(RUNS, "decision_struct.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"base": base_t, "base_acc": base_acc,
                   "results": results}, f, ensure_ascii=False, indent=2)
    print(f"  结果已写入 {out}   耗时 {time.time()-T0:.1f}s")


if __name__ == "__main__":
    main()
