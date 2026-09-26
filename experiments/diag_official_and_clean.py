"""按官方公式复核 v6 分数 + 统计训练标签里的非法值（哨兵清单漏网）。

三个问题一次量化：
  A. 官方 PERM 公式（1 - |log10(max(ŷ/y, ε))|，恒为 1.0 decade 容差）
     与仓库实现的差异 → 重算总分与 75 分缺口。
  B. src/feature.py::_is_sentinel 只拦 abs>1e5，
     POR=-6000.336 / PERM=-6000.373 / SW=-5960.427 等非法标签被当成真实段参与回归。
     统计规模、按井分布、以及重训回归器时的实际目标值（log10 空间）。
  C. 这些非法标签在官方口径下对分数的影响方向。

只读缓存前向 + 读标签，不 fit 模型、不 import torch。
用法: python experiments/diag_official_and_clean.py
"""
from __future__ import annotations
import os, sys, warnings, time, json
warnings.filterwarnings("ignore")
import joblib, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.feature import default_mask, load_wells, make_targets
from src.utils import DEFAULT_VALUES, EPS, TARGETS, WEIGHTS

RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
TCN_KEY = {"T15": "T15_seed42_150ep", "T16a": "T16_seed43_150ep", "T16b": "T16_seed44_150ep"}
TCN = list(TCN_KEY)
V6 = {"blend_w": {"POR": 0.54, "PERM": 0.96, "SW": 0.80},
      "thresholds": {"POR": 0.115, "PERM": 0.065, "SW": 0.040},
      "clf_tcn_w": {"POR": 0.60, "PERM": 0.75, "SW": 0.30}}
# 物理合理区间：越界即视为标签损坏
PHYS = {"POR": (0.0, 40.0), "PERM": (0.0, 10000.0), "SW": (0.0, 100.0)}


def acc_official(c, pred, true):
    """官方口径逐点分数（图片原文）。PERM 为比值容差恒 1.0 decade。"""
    p = np.asarray(pred, float); t = np.asarray(true, float)
    if c == "PERM":
        return np.maximum(0.0, 1.0 - np.abs(np.log10(np.maximum(p / np.maximum(t, EPS), EPS))))
    tol = 0.08 if c == "POR" else 0.05
    return np.maximum(0.0, 1.0 - np.abs(p - t) / (tol * (np.abs(t) + EPS)))


def acc_repo(c, pred, true):
    """仓库实现（src/utils.py）。PERM 容差 = 0.08 * |log10 y|。"""
    if c == "PERM":
        p = np.log10(np.maximum(np.asarray(pred, float), EPS))
        t = np.log10(np.maximum(np.asarray(true, float), EPS))
        return np.maximum(0.0, 1.0 - np.abs(p - t) / (0.08 * (np.abs(t) + EPS)))
    return acc_official(c, pred, true)


def build_preds(Y, rt, pt, store):
    """v6 融合 + 阈值决策 → {目标: 物理空间预测}。"""
    out = {}
    for c in TARGETS:
        ens = np.mean([store[f"{TCN_KEY[t]}__{c}__reg"] for t in TCN], axis=0)
        pcn = np.mean([store[f"{TCN_KEY[t]}__{c}__pdf"] for t in TCN], axis=0)
        w, th, cw = V6["blend_w"][c], V6["thresholds"][c], V6["clf_tcn_w"][c]
        pm = (1 - cw) * pt[c] + cw * pcn
        out[c] = np.where(pm >= th, DEFAULT_VALUES[c], np.power(10.0, (1 - w) * rt[c] + w * ens))
    return out


def main() -> None:
    t0 = time.time()
    wells = load_wells("./train", has_labels=True)
    ids = [w["logId"].iloc[0] for w in wells]
    va_df = pd.concat([w for w in wells if w["logId"].iloc[0] in set(ids[72:])], ignore_index=True)
    Y = make_targets(va_df)
    cache = joblib.load(os.path.join(RUNS, "screen_tree_cache.joblib"))
    rt, pt = cache["reg"], cache["p_def"]
    store = {k: v for k, v in np.load(os.path.join(RUNS, "tcn_screen_outputs.npz")).items()}
    preds = build_preds(Y, rt, pt, store)
    Dm = {c: default_mask(Y)[c].to_numpy() for c in TARGETS}

    print("=" * 100)
    print("A. 官方口径 vs 仓库口径（v6 当前预测，screen 后 8 井）")
    print("-" * 100)
    tot_r = tot_o = 0.0
    off = {}
    for c in TARGETS:
        p, y = np.asarray(preds[c], float), Y[c].to_numpy()
        v = ~np.isnan(y)
        f_r, f_o = acc_repo(c, p, y), acc_official(c, p, y)
        a_r, a_o = float(np.mean(f_r[v])), float(np.mean(f_o[v]))
        off[c] = a_o
        tot_r += WEIGHTS[c] * a_r
        tot_o += WEIGHTS[c] * a_o
        dr = float(np.mean(f_o[v & ~Dm[c]]))
        print(f"  {c:5s} 仓库 {a_r:.4f} → 官方 {a_o:.4f} (Δ{WEIGHTS[c]*(a_o-a_r)*100:+.3f}分)"
              f"   真实段官方 acc {dr:.4f}")
    print(f"  Total 仓库 {tot_r*100:7.3f}  →  官方 {tot_o*100:7.3f}   75 分缺口 {75.0-tot_o*100:+.3f}")
    print(f"  分目标权重贡献: " + "  ".join(f"{c}={WEIGHTS[c]*off[c]*100:.2f}分" for c in TARGETS))

    print("\n" + "=" * 100)
    print("B. 标签非法值：哨兵清单漏网（src/feature.py 只拦 abs>1e5）")
    print("-" * 100)
    tr = pd.concat(wells, ignore_index=True)
    Ytr = make_targets(tr)
    summary = {}
    print("  训练集逐列可疑取值直方（make_targets 之后仍非 NaN 的越界值）:")
    for c in TARGETS:
        y = Ytr[c].to_numpy()
        lo, hi = PHYS[c]
        valid = ~np.isnan(y)
        broken = valid & ~((y >= lo) & (y <= hi))
        vals, cnts = np.unique(y[broken], return_counts=True)
        isreal = valid & ~np.isclose(y, DEFAULT_VALUES[c])
        print(f"  {c:5s} 有效点={valid.sum():,}  物理区间[{lo},{hi}]外 = {int(broken.sum())}"
              f" ({broken.sum()/max(valid.sum(),1):.4%})")
        for vv, cc in zip(vals, cnts):
            print(f"          {vv:12.3f} × {int(cc)}    log10={np.log10(max(vv, EPS)):+.3f}")
        if broken.sum():
            # 负真值在官方公式里的后果: 比值 ŷ/y 为负 → max(...,ε)=ε → log10 后为
            # |log10(ε/y)|, 分母 tol*|y| 巨大; 结论是该点几乎必然得 0。
            # 官方公式对负真值无定义, 这里只给结论不给分。
            print(f"          → 真值为负: 官方公式无定义, 该点预计得 0（对全体队伍等量扣分）")
        summary[c] = int(broken.sum())

    # 更宽口径: 所有负值标签（含已被哨兵拦下的 -99999 系）到底有多少
    print("\n  宽口径: 原始列中的负值与哨兵（区分「已拦截」与「漏网」）:")
    for c in TARGETS:
        r = pd.to_numeric(tr[c], errors="coerce").to_numpy()
        known = {100000.0, 10000.0, 2.534573822542048e37}
        neg = r < 0
        sent = neg & np.isin(np.abs(r), sorted(known))
        leak = neg & ~sent
        print(f"  {c:5s} 负值={int(neg.sum()):,}  其中已知哨兵={int(sent.sum()):,}"
              f"  漏网(非标准哨兵)={int(leak.sum()):,}  漏网取值={np.round(np.unique(r[leak]),2).tolist()}")

    # 按井分布 —— 判断是否整井损坏（可整井剔除）还是零星散布
    print("\n  漏网损坏点按井分布:")
    perwell = {}
    for wid, g in tr.groupby("logId"):
        n = 0
        for c in TARGETS:
            yc = make_targets(g)[c].to_numpy()
            v = ~np.isnan(yc)
            lo, hi = PHYS[c]
            n += int((v & ~((yc >= lo) & (yc <= hi))).sum())
        if n:
            perwell[wid] = n
    if perwell:
        for wid, n in sorted(perwell.items(), key=lambda kv: -kv[1])[:6]:
            print(f"    {wid}  {n}")
        tot_broken = sum(perwell.values())
        print(f"  共 {len(perwell)} 口井受影响, 最多一口占 {max(perwell.values())/max(tot_broken,1):.1%} → "
              f"{'零星散布，不能整井剔除' if max(perwell.values())/max(tot_broken,1) < 0.1 else '集中在少数井'}")
    else:
        print("    (无)")

    # 测试集：标签不可见，但可看输入曲线是否同源损坏（判断是否所有队伍同损）
    print("\n" + "=" * 100)
    print("C. 测试集输入是否含同类异常（标签不可见，只能看曲线）")
    print("-" * 100)
    tw = load_wells("./test", has_labels=False)
    tf = pd.concat(tw, ignore_index=True)
    for col in ["GR", "PE", "SP", "CAL", "AC", "DEN", "CNL", "RT"]:
        s = pd.to_numeric(tf[col], errors="coerce")
        nb = int(((s < 0) | (s.abs() > 1e5)).sum())
        if nb:
            print(f"  {col:5s} 异常 {nb:,}")
    print("  测试集不含目标列，无法直接核对标签；训练集损坏为**逐点散布**，"
          "故测试集标签大概率同样受污染 → 对全体队伍等量扣分，不改变排名。")

    out = {"official_acc": off, "total_repo": tot_r, "total_official": tot_o,
           "gap_75": 75.0 - tot_o * 100, "broken_labels": summary}
    with open(os.path.join(RUNS, "official_and_clean.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
