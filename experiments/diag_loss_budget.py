"""官方测试集已知 75.73 之后，重算「还有多少分可追、在谁身上」。

之前 §4.2 说"分段器可追 +2.997、回归器到顶"，但那个结论没有把丢分拆到
「分段器」和「回归器」两个独立的因子上。本脚本做这个分解（纯缓存，约 5 秒，不训练）：

  部署决策   ：p_def >= th → 默认值，否则 → 树/TCN 融合回归
  真值分段   ：真段 → 融合回归，默认段 → 默认值          ← 分段器完美时的天花板
  段内池化   ：真段 → 该段融合回归的**中位数**，默认段 → 默认值  ← 回归去噪的天花板

三者的差就是两条互不相干的账：分段器丢掉多少、回归器丢掉多少。
另外报真实段的回归质量（corr(log10)、比值中位），判断"回归器差"是系统性偏置还是噪声。

用法: python experiments/diag_loss_budget.py
"""
from __future__ import annotations
import os, sys, time
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


def acc(c, pred, true):
    p = np.asarray(pred, np.float64); t = np.asarray(true, np.float64)
    if c == "PERM":
        return np.maximum(0.0, 1.0 - np.abs(np.log10(np.maximum(p / np.maximum(t, EPS), EPS))))
    tol = 0.08 if c == "POR" else 0.05
    return np.maximum(0.0, 1.0 - np.abs(p - t) / (tol * (np.abs(t) + EPS)))


def median_filter(arr, k):
    """居中窗口中值滤波（沿深度，不跨点）。k 为奇数。"""
    if k <= 1:
        return arr.copy()
    half = k // 2
    n = len(arr)
    out = np.empty(n, np.float64)
    pad = np.pad(arr, half, mode="edge")
    for i in range(n):
        out[i] = np.median(pad[i:i + k])
    return out


def main() -> None:
    st = joblib.load(os.path.join(RUNS, "screen_tree_cache.joblib"))
    Z = np.load(os.path.join(RUNS, "tcn_all80_parts.npz"))
    ens = {c: np.float32(Z[f"reg_{c}"][BND:]) for c in TARGETS}
    pdf = {c: np.float32(Z[f"pdf_{c}"][BND:]) for c in TARGETS}

    wells = load_wells("./train", has_labels=True)
    ids = [w["logId"].iloc[0] for w in wells]
    va = [w for w in wells if w["logId"].iloc[0] in set(ids[72:])]
    df = pd.concat(va, ignore_index=True)
    Y = make_targets(df); DM = default_mask(Y)

    tot = {"deployed": 0.0, "true_seg": 0.0, "pool": 0.0, "pool25": 0.0}
    print("=" * 100)
    print("screen 8 井：丢分归因（部署 v7 vs 真值分段 vs 段内池化）")
    print("=" * 100)
    for c in TARGETS:
        y = Y[c].to_numpy()
        m = ~np.isnan(y)
        dflt = DM[c].to_numpy().astype(bool)
        real = ~dflt
        rm = m & real
        pm = (1 - V7["clf_tcn_w"][c]) * st["p_def"][c] + V7["clf_tcn_w"][c] * pdf[c]
        regb = np.power(10.0, (1 - V7["blend_w"][c]) * np.asarray(st["reg"][c], np.float64)
                        + V7["blend_w"][c] * ens[c])
        A = lambda p: float(acc(c, p[m], y[m]).mean())
        a_dep = A(np.where(pm >= V7["thresholds"][c], DEFAULT_VALUES[c], regb))
        a_true = A(np.where(~real, DEFAULT_VALUES[c], regb))

        # 段内池化：真段内取点级预测的池化值（中值）。
        # 段 id：cumsum(real) 会跨非储层间隙保持常数，所以非储层行单独标 -1
        seg = np.cumsum(real.astype(int))
        seg[~real] = -1
        med = pd.Series(np.where(real, regb, np.nan)).groupby(seg).transform("median")
        r_seg = np.where(real, med.to_numpy(), regb)
        a_pool = {
            "点级": A(np.where(~real, DEFAULT_VALUES[c], regb)),
            "中值k5": A(np.where(~real, DEFAULT_VALUES[c], median_filter(regb, 5))),
            "中值k15": A(np.where(~real, DEFAULT_VALUES[c], median_filter(regb, 15))),
            "段内中值": A(np.where(~real, DEFAULT_VALUES[c], r_seg)),
        }
        tot["deployed"] += WEIGHTS[c] * a_dep * 100
        tot["true_seg"] += WEIGHTS[c] * a_true * 100
        tot["pool"] += WEIGHTS[c] * a_pool["段内中值"] * 100

        fr = m & dflt & (pm < V7["thresholds"][c])   # 真默认被误判为真实
        fd = m & real & (pm >= V7["thresholds"][c])  # 真真实被误判为默认
        cc = float(np.corrcoef(np.log10(np.maximum(regb[rm], EPS)),
                               np.log10(np.maximum(y[rm], EPS)))[0, 1])
        print(f"\n{c}  默认点 {(m & dflt).sum():,} / 真点 {rm.sum():,}")
        print(f"  部署 acc      {a_dep:.4f}   (贡献 {WEIGHTS[c]*a_dep*100:6.3f})")
        print(f"  真值分段 acc  {a_true:.4f}   (贡献 {WEIGHTS[c]*a_true*100:6.3f})"
              f"   ← 分段器可追 {WEIGHTS[c]*(a_true-a_dep)*100:+.3f}")
        print(f"    其中误判：真默认→判真 {fr.sum():,} ；真真实→判默认 {fd.sum():,}")
        print(f"  真实段回归质量：real_acc {acc(c, regb[rm], y[rm]).mean():.4f} ，"
              f"corr(log10 回归, log10 真值) {cc:+.3f} ，"
              f"回归/真值 比值中位 {np.median(regb[rm] / np.maximum(y[rm], EPS)):.3f}")
        print(f"  真实段去噪天花板：点级 {a_pool['点级']:.4f} → 中值k5 {a_pool['中值k5']:.4f} "
              f"→ 中值k15 {a_pool['中值k15']:.4f} → 段内中值 {a_pool['段内中值']:.4f}")
        tot["pool25"] += WEIGHTS[c] * a_pool["中值k15"] * 100

    print("\n" + "=" * 100)
    print(f"部署（v7）        = {tot['deployed']:.3f}   （harness 复算，应与 73.749 一致）")
    print(f"真值分段 + 同回归 = {tot['true_seg']:.3f}   分段器可追 {tot['true_seg']-tot['deployed']:+.3f}")
    print(f"真值分段 + 段内中值回归 = {tot['pool']:.3f}   回归去噪可追 "
          f"{tot['pool']-tot['true_seg']:+.3f}")
    print(f"真值分段 + 中值k15 回归 = {tot['pool25']:.3f}   回归去噪可追 "
          f"{tot['pool25']-tot['true_seg']:+.3f}")
    print(f"\n两段相加 = 真值分段+段内中值 - 部署 = {tot['pool']-tot['deployed']:+.3f}"
          f"  （相对 75 线缺口 +1.251，官方测试集实测 75.73）")
    print(f"耗时 {time.time()-T0:.1f}s")


if __name__ == "__main__":
    main()
