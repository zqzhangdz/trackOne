"""Track: 截断非凸指标下的集成决策 (mean vs median vs routing vs calibration).

只用缓存前向 + 真值分段, 不 fit 任何新模型, 不 import torch.
成员 = tree(72井训练) + T15/T16a/T16b, 与 v9 部署成员同构.

对 max(0, 1 - e/tol) 这种带外零梯度指标, 逐点凸平均是否最优?
  H1 median-of-4 优于 mean-of-4 (双峰平均落入间隙)
  H2 逐目标 log10 仿射校准 a+b*blend 修正系统性偏差 (PERM 预测/真值中位 0.671)
  H3 代理中心 best-of-N 路由优于凸平均
  H4 成员间不一致度 = 路由可用空间的量化

评估: 8 口 screen 井, 官方口径, 逐点等权.
无拟合参数的比较 (H1/H3/H4) 直接评估; 需拟合的 (H2) 用留一井拟合.
"""
from __future__ import annotations
import os, sys, gc, time, warnings
warnings.filterwarnings("ignore")
import joblib, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.feature import load_wells, make_targets, default_mask
from src.utils import DEFAULT_VALUES, EPS, TARGETS

RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
MEM = ["T15_seed42_150ep", "T16_seed43_150ep", "T16_seed44_150ep"]
TOL = {"POR": 0.08, "PERM": None, "SW": 0.05}
WTS = {  # 与 models/model.pkl 完全一致的 4 元权重向量
    "POR": [0.5, 1 / 6, 1 / 6, 1 / 6],
    "PERM": [0.5, 1 / 6, 1 / 6, 1 / 6],
    "SW": [0.25, 0.25, 0.25, 0.25],
}


def acc_vec(c, pred, true):
    p = np.asarray(pred, float); t = np.asarray(true, float)
    if c == "PERM":
        return np.maximum(0.0, 1.0 - np.abs(np.log10(np.maximum(p / np.maximum(t, EPS), EPS))))
    return np.maximum(0.0, 1.0 - np.abs(p - t) / (TOL[c] * (np.abs(t) + EPS)))


def rel_err(c, pred, true):
    """官方口径归一化误差 (PERM = |log10 ratio|, 其余 = 相对误差)."""
    p = np.asarray(pred, float); t = np.asarray(true, float)
    if c == "PERM":
        return np.abs(np.log10(np.maximum(p / np.maximum(t, EPS), EPS)))
    return np.abs(p - t) / (TOL[c] * (np.abs(t) + EPS))


def main() -> None:
    t0 = time.time()
    wells = load_wells("./train", has_labels=True)
    ids = [w["logId"].iloc[0] for w in wells]
    va = pd.concat([w for w in wells if w["logId"].iloc[0] in set(ids[72:])], ignore_index=True)
    del wells; gc.collect()
    Y = make_targets(va)
    DM = default_mask(Y)

    cache = joblib.load(os.path.join(RUNS, "screen_tree_cache.joblib"))
    rt, va_ids = cache["reg"], cache["va_ids"]
    store = {k: v for k, v in np.load(os.path.join(RUNS, "tcn_screen_outputs.npz")).items()}
    well_of = va["logId"].to_numpy()

    M = {c: np.array([rt[c]] + [store[f"{t}__{c}__reg"] for t in MEM]) for c in TARGETS}  # (4,N) log10
    y = {c: Y[c].to_numpy() for c in TARGETS}
    # DM[c]=True 表示默认段 (isclose(y, 默认常数)); 我们只关心储层段 = ~DM
    d = {c: (~DM[c].to_numpy()) & ~np.isnan(y[c]) for c in TARGETS}  # 真实储层段掩码

    def score(pred_phys, c):
        a = acc_vec(c, pred_phys, y[c])
        return a.mean(), a[d[c]].mean(), a[d[c]].size

    # ------------------------------------------------------------------
    print("=" * 84)
    print("A. 成员结构与 baseline (真值分段掩码 → 隔离回归器质量, 与分段器无关)")
    print("=" * 84)
    for c in TARGETS:
        w = np.array(WTS[c])
        per = np.power(10.0, w @ M[c])
        pred = np.where(d[c], per, DEFAULT_VALUES[c])
        tot, real, n = score(pred, c)
        e = rel_err(c, per, y[c])[d[c]]
        print(f"  {c:5s} 储层段 n={n:,}  凸平均 acc={tot:.4f}  reservoir acc={real:.4f}"
              f"  带内率={float((e < 1.0).mean()):.4f}  带外点均误差={e[e>=1.0].mean() if (e>=1.0).any() else 0:.2f}")
        off = np.corrcoef(M[c])[np.triu_indices(4, 1)]
        print(f"        成员两两 log10 相关: min={off.min():.4f}  mean={off.mean():.4f}  max={off.max():.4f}")
        print(f"        成员 log10 均值: tree={M[c][0].mean():+.4f}  "
              + "  ".join(f"{MEM[i].split('_')[0]}={M[c][i+1].mean():+.4f}" for i in range(3)))

    # ------------------------------------------------------------------
    print("\n" + "=" * 84)
    print("B. H1: 中位数-of-4 vs 凸平均 (无拟合参数)")
    print("=" * 84)
    for c in TARGETS:
        w = np.array(WTS[c])
        for name, agg in [("凸平均 ", lambda X, w=w: w @ X), ("等权平均", lambda X, w=w: X.mean(axis=0)),
                          ("中位数 ", lambda X, w=w: np.median(X, axis=0)),
                          ("均值上取", lambda X, w=w: np.max(X, axis=0)),
                          ("均值下取", lambda X, w=w: np.min(X, axis=0))]:
            pred = np.where(d[c], np.power(10.0, agg(M[c])), DEFAULT_VALUES[c])
            tot, real, _ = score(pred, c)
            print(f"  {c:5s} {name} reservoir acc={real:.4f}  total acc={tot:.4f}")

    # ------------------------------------------------------------------
    print("\n" + "=" * 84)
    print("C. H2: 逐目标 log10 仿射校准 a + b*blend (留一井拟合, 无泄漏)")
    print("=" * 84)
    for c in TARGETS:
        w = np.array(WTS[c])
        bl = w @ M[c]
        base_pred = np.where(d[c], np.power(10.0, bl), DEFAULT_VALUES[c])
        tot0, real0, _ = score(base_pred, c)

        # 目标空间: PERM 拟合 log10(y); 其余也拟合 log10(y) (乘性偏差在 log 空间是加性的)
        ty = np.log10(np.maximum(y[c], EPS))
        variants = {}
        for vname, sel in [
            ("全储层", d[c]),
            ("带内", d[c] & (rel_err(c, np.power(10.0, bl), y[c]) < 1.0)),
            ("带内中位锚", d[c] & (rel_err(c, np.power(10.0, bl), y[c]) < 0.5)),
            ("仅比例=1 (b锁死)", d[c]),
        ]:
            variants[vname] = sel

        rows = []
        for vname, sel in variants.items():
            a_loo, r_loo, per_well = [], [], []
            for iid in va_ids:
                mw = well_of == iid
                tr = np.ones(len(bl), bool); tr[mw] = False
                fi = sel & tr
                if vname == "仅比例=1 (b锁死)":
                    b = 1.0
                    rr = np.power(10.0, bl)[fi] / np.maximum(y[c][fi], EPS)
                    a = np.log10(np.median(rr))
                else:
                    if fi.sum() < 30:
                        continue
                    a, b = np.polyfit(bl[fi], ty[fi], 1)
                pred = np.where(d[c], np.power(10.0, a + b * bl), DEFAULT_VALUES[c])
                aa = acc_vec(c, pred, y[c])
                a_loo.append(aa[mw].mean())
                r_loo.append(aa[d[c] & mw].mean())
                per_well.append(aa[mw].mean() - acc_vec(c, base_pred, y[c])[mw].mean())
            if len(a_loo) == 0:
                continue
            rows.append((vname, a, b, np.mean(a_loo), np.mean(r_loo),
                         min(per_well), max(per_well), sum(p > 0 for p in per_well)))
        print(f"  {c:5s} 基线(凸平均) total={tot0:.4f}  reservoir={real0:.4f}")
        for vname, a0, b0, tl, rl, lo, hi, pos in rows:
            print(f"        {vname:12s} LOO total={tl:.4f} reservoir={rl:.4f} Δtotal={tl-tot0:+.4f}"
                  f"  逐井Δ=[{lo:+.4f},{hi:+.4f}] 正号{pos}/8")
            print(f"        {'':12s} a={a0:+.5f} b={b0:.5f}  → 乘性修正 {10**a0:.4f}x"
                  + ("  (b=1 锁定, 纯偏差修正)" if vname == "仅比例=1 (b锁死)" else ""))

    # ------------------------------------------------------------------
    print("\n" + "=" * 84)
    print("D. 系统性乘性偏差 (预测/真值 比, 真值储层段) —— 校准的可操作空间")
    print("=" * 84)
    for c in TARGETS:
        w = np.array(WTS[c])
        bl = w @ M[c]
        per = np.power(10.0, bl)
        rr = per[d[c]] / np.maximum(y[c][d[c]], EPS)
        print(f"  {c:5s} pred/true: 中位={np.median(rr):.4f}  均值={np.mean(rr):.3f}"
              f"  p10={np.quantile(rr,.1):.3f} p25={np.quantile(rr,.25):.3f}"
              f"  p75={np.quantile(rr,.75):.3f} p90={np.quantile(rr,.9):.3f}")
        print(f"        |log10 ratio|: 中位={np.median(np.abs(np.log10(rr))):.4f}"
              f"  均值={np.abs(np.log10(rr)).mean():.4f}  → 若全部修正到 1.0, reservoir acc 上限提升可观")

    # ------------------------------------------------------------------
    print("\n" + "=" * 84)
    print("E. H3: 代理中心 best-of-N 路由 (无拟合参数, 代理只用成员输出)")
    print("=" * 84)
    print("  硬选 = argmin_j |member_j - proxy| ; 对照 = 凸平均 / 中位数")
    for c in TARGETS:
        w = np.array(WTS[c])
        ref = acc_vec(c, np.where(d[c], np.power(10.0, w @ M[c]), DEFAULT_VALUES[c]), y[c])
        proxies = {
            "tree": M[c][0],
            "TCN3均值": M[c][1:].mean(axis=0),
            "中位4": np.median(M[c], axis=0),
            "等权均值4": M[c].mean(axis=0),
        }
        for pn, pr in proxies.items():
            j = np.argmin(np.abs(M[c] - pr[None, :]), axis=0)
            pred = np.where(d[c], np.power(10.0, M[c][j]), DEFAULT_VALUES[c])
            aa = acc_vec(c, pred, y[c])
            print(f"  {c:5s} proxy={pn:9s} reservoir={aa[d[c]].mean():.4f} total={aa.mean():.4f}"
                  f"  Δtotal={aa.mean()-ref.mean():+.4f}")
        # 成员被选中的频率
        j = np.argmin(np.abs(M[c] - proxies['等权均值4'][None, :]), axis=0)
        print(f"        成员被选中频率(等权均值proxy): " +
              ", ".join(f"{n}={float((j==k).mean()):.3f}" for k, n in enumerate(["tree"] + MEM)))

    # ------------------------------------------------------------------
    print("\n" + "=" * 84)
    print("F. H4: 成员间不一致度 —— 路由的理论可用空间")
    print("=" * 84)
    for c in TARGETS:
        e = rel_err(c, np.power(10.0, M[c]), y[c])  # (4,N) 每成员的官方归一化误差
        spread = e.max(axis=0) - e.min(axis=0)
        mix = spread[d[c]] > 1.0            # 至少一个成员带内、另一个带外
        print(f"  {c:5s} 储层段 n={int(d[c].sum()):,}")
        print(f"        成员误差 spread 中位={np.median(spread[d[c]]):.4f}  "
              f"p50/p90={np.quantile(spread[d[c]],.5):.4f}/{np.quantile(spread[d[c]],.9):.4f}")
        print(f"        混叠点占比 (spread>1, 4成员间至少一个能进带): {mix.mean():.4f}"
              f"  <-- 无拟合路由的分数上限来源")
        # 4成员都带外的点占比 (完全救不了)
        allout = np.all(e >= 1.0, axis=0)
        print(f"        4成员全带外占比: {allout[d[c]].mean():.4f}  至少一个带内: {float((~allout)[d[c]].mean()):.4f}")
        # 完美成员选择器 (知道真值) 的分数 = 逐点取 4 成员最优
        sc = np.maximum(0.0, 1.0 - e)   # (4,N) 每成员的逐点官方分数
        oracle = sc.max(axis=0)
        bl = np.array(WTS[c]) @ M[c]
        base = acc_vec(c, np.power(10.0, bl), y[c])
        print(f"        [神谕上限] 逐点取4成员中官方分数最高者: reservoir acc={oracle[d[c]].mean():.4f}"
              f"  vs 凸平均 {base[d[c]].mean():.4f}  → 成员池内路由的理论上限 Δ={oracle[d[c]].mean()-base[d[c]].mean():+.4f}")

    print(f"\n耗时 {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
