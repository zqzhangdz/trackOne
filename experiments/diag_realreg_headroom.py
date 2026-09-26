"""真实段回归器余量：在**真值分段**下，POR/SW 的逐点回归还有多少可挤？

动机
----
§4.2 用 PERM real acc 0.51 判定"回归器基本没问题"，但 POR real acc 只有 0.1694、SW 0.1450，
而**从未有人测过 POR/SW 真实段回归自己的天花板**。若真实段回归还有 0.1 的余量，那就是
+1.3 分 —— 与分段器可追的 +2.587 并列的第二块。§4.6 只测过 PERM 的 loss/加权，
§7 的"推理后平滑"是按**旧 PERM 口径**否决的（当时阈值很低、真实段占比高，平滑会混入
默认段常数）。官方口径下（阈值高、默认段占比 0.60–0.73）从未复验。

关键判据：真实段的回归**残差是否自相关**？
  * 自相关高 → 残差是系统性偏差 → 平滑 / 向段均值收缩可降误差 → **有可部署余量**；
  * 自相关 ≈ 0 → 残差是白噪声，8%/5% 相对容差就是硬墙 → **无余量**。

纯缓存前向，不 fit、不 import torch（§5.1）。用真值分段是**不可达上界**下的机制诊断，
不是可达分数 —— 见输出末尾的口径说明。

用法: python experiments/diag_realreg_headroom.py
"""
from __future__ import annotations
import os, sys, gc, time, warnings
warnings.filterwarnings("ignore")
import joblib, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.feature import load_wells, make_targets, default_mask
from src.utils import DEFAULT_VALUES, EPS, TARGETS

RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
T108 = {"T15": "T15_seed42_150ep", "T16a": "T16_seed43_150ep", "T16b": "T16_seed44_150ep"}
# v7 blend_w：reg = (1-w)*树 + w*TCN均值（log10 空间）
BLEND = {"POR": 0.50, "PERM": 0.50, "SW": 0.75}
TOL = {"POR": 0.08, "PERM": 1.0, "SW": 0.05}


def acc_vec(c, pred, true):
    p = np.asarray(pred, float); t = np.asarray(true, float)
    if c == "PERM":
        return np.maximum(0.0, 1.0 - np.abs(np.log10(np.maximum(p / np.maximum(t, EPS), EPS))))
    return np.maximum(0.0, 1.0 - np.abs(p - t) / (TOL[c] * (np.abs(t) + EPS)))


def rel_err(c, pred, true):
    """归一化到「容差单位」的相对误差：>1 即完全失分。"""
    p = np.asarray(pred, float); t = np.asarray(true, float)
    if c == "PERM":
        return np.abs(np.log10(np.maximum(p / np.maximum(t, EPS), EPS)))
    return np.abs(p - t) / (TOL[c] * (np.abs(t) + EPS))


def run_ids(flag):
    """把连续 True 段编号（0,1,2,...），False 位置为 -1。"""
    ids = np.full(len(flag), -1, dtype=np.int64)
    n = 0
    for i in range(1, len(flag)):
        if flag[i] and not flag[i - 1]:
            n += 1
        if flag[i]:
            ids[i] = n - 1
    return ids


def smooth_in_runs(raw, flag, width):
    """仅在真实段**连续段内**做居中滑动平均，段长 < width 时回退原值。"""
    out = raw.copy()
    ids = run_ids(flag)
    for rid in np.unique(ids):
        if rid < 0:
            continue
        sl = np.flatnonzero(ids == rid)
        if len(sl) < width:
            continue
        v = raw[sl]
        k = (width - 1) // 2
        # 端点复制 padding 后 'valid' 卷积：输出长度 = (n2+2k) - width + 1 = n2。
        # 注意必须用 np.repeat —— `v[:1] * k` 是标量乘法，长度恒为 1，pad 只多 2 个元素。
        pad = np.concatenate([np.repeat(v[:1], k), v, np.repeat(v[-1:], k)])
        out[sl] = np.convolve(pad, np.ones(width) / width, mode="valid")
    return out


def main() -> None:
    t0 = time.time()
    wells = load_wells("./train", has_labels=True)
    ids = [w["logId"].iloc[0] for w in wells]
    va = pd.concat([w for w in wells if w["logId"].iloc[0] in set(ids[72:])], ignore_index=True)
    del wells; gc.collect()
    va = va.sort_index().reset_index(drop=True)
    Y = make_targets(va)
    DM = default_mask(Y)
    assert len(va) == 72779, len(va)

    cache = joblib.load(os.path.join(RUNS, "screen_tree_cache.joblib"))
    rt, pt = cache["reg"], cache["p_def"]
    store = {k: v for k, v in np.load(os.path.join(RUNS, "tcn_screen_outputs.npz")).items()}

    depth_ok = True
    for w in va["logId"].unique():
        d = va.loc[va["logId"] == w, "DEPTH"].to_numpy()
        if not np.all(np.diff(d) > 0):
            depth_ok = False
            break

    # 融合回归（log10 空间）→ 原空间。缓存三个目标**全在 log10**（见 seq_clf_probe.log10ify）。
    pred, y_true, is_real = {}, {}, {}
    for c in TARGETS:
        ens = np.mean([store[f"{T108[t]}__{c}__reg"] for t in T108], axis=0)
        reg = (1 - BLEND[c]) * rt[c] + BLEND[c] * ens
        pred[c] = np.power(10.0, np.where(np.isfinite(reg), reg, -5.0))
        y_true[c] = Y[c].to_numpy().astype(float)
        # DM 为 True 表示「值 == 默认值」，所以真实段是 ~DM。
        is_real[c] = (~DM[c].to_numpy()) & np.isfinite(y_true[c])

    print("=" * 82)
    print(f"真值分段下的真实段回归诊断  |  验证 {len(va):,} 行  |  深度升序单调: {depth_ok}")
    print("=" * 82)
    print(f"{'目标':6s}{'真值分段占比':>12s}{'真实段点数':>11s}{'acc_real':>10s}"
          f"{'|err|中位':>11s}{'|err|p90':>10s}{'真值中位':>10s}")
    for c in TARGETS:
        m = is_real[c]
        a = acc_vec(c, pred[c][m], y_true[c][m])
        e = rel_err(c, pred[c][m], y_true[c][m])
        print(f"{c:6s}{m.mean():>12.4f}{m.sum():>11,d}{a.mean():>10.4f}"
              f"{np.median(e):>11.3f}{np.quantile(e,0.9):>10.3f}{np.median(y_true[c][m]):>10.3f}")
    print(f"[自检] 真值分段 + 融合回归：POR acc_real 应 ≈ 0.2376（§4.2 表反推）")

    # ── 核心：残差是否自相关 ──
    print("\n" + "=" * 82)
    print("[核心] 真实段残差自相关（同段内，lag = 深度间隔 0.1 m × k）")
    print("  自相关越高 → 误差是系统性偏差 → 平滑/收缩可降误差 → 有可部署余量")
    for c in TARGETS:
        m = is_real[c]
        ids_run = run_ids(m)
        err = rel_err(c, pred[c], y_true[c])
        line = f"  {c:5s}"
        for k in (1, 2, 3, 5, 10, 20, 50):
            same = (ids_run[:-k] == ids_run[k:]) & (ids_run[:-k] >= 0)
            a1 = err[:-k][same]; a2 = err[k:][same]
            r = float(np.corrcoef(a1, a2)[0, 1]) if len(a1) > 50 else float("nan")
            line += f"  lag{k:<3d}{r:>7.3f}"
        print(line)

    # ── 实验：段内平滑 / 向段均值收缩 / 段常数 ──
    print("\n" + "=" * 82)
    print("[实验] 段内处理对 acc_real 的影响（真值分段，不可达上界下的机制诊断）")
    print(f"{'目标':6s}{'raw':>8s}{'MA3':>8s}{'MA5':>8s}{'MA11':>8s}{'MA21':>8s}"
          f"{'→段均值':>9s}{'段常数':>9s}")
    for c in TARGETS:
        m = is_real[c]
        base = float(acc_vec(c, pred[c][m], y_true[c][m]).mean())
        row = f"{c:6s}{base:>8.4f}"
        for w in (3, 5, 11, 21):
            p = smooth_in_runs(pred[c], m, w)
            row += f"{float(acc_vec(c, p[m], y_true[c][m]).mean()):>8.4f}"
        # 向「所在连续段的预测均值」收缩，最优 α
        best_a, best_s = 0.0, base
        p_full = pred[c].copy()
        ids_run = run_ids(m)
        for alpha in np.linspace(0.0, 1.0, 21):
            q = p_full.copy()
            for rid in np.unique(ids_run):
                if rid < 0:
                    continue
                sl = np.flatnonzero(ids_run == rid)
                q[sl] = (1 - alpha) * p_full[sl] + alpha * p_full[sl].mean()
            s = float(acc_vec(c, q[m], y_true[c][m]).mean())
            if s > best_s:
                best_a, best_s = alpha, s
        row += f"{best_s:>9.4f}"
        # 段常数（每段取该段预测中位数）
        q = p_full.copy()
        for rid in np.unique(ids_run):
            if rid < 0:
                continue
            sl = np.flatnonzero(ids_run == rid)
            q[sl] = np.median(p_full[sl])
        row += f"{float(acc_vec(c, q[m], y_true[c][m]).mean()):>9.4f}"
        print(row + f"   (最优收缩 α={best_a:.2f})")

    # ── 偏置检查 ──
    print("\n" + "=" * 82)
    print("[偏置] 真实段上的系统性偏差（若明显，乘法/加法校正就有白捡的分）")
    for c in TARGETS:
        m = is_real[c]
        ratio = pred[c][m] / np.maximum(y_true[c][m], EPS)
        print(f"  {c:5s} pred/true 中位 {np.median(ratio):>8.3f}  "
              f"p10 {np.quantile(ratio,0.1):>7.3f}  p90 {np.quantile(ratio,0.9):>7.3f}")

    # ── 换算成分 ──
    print("\n" + "=" * 82)
    print("[换算] 若真实段 acc_real 提升 Δ，总分提升 = Δ × 真实段占比 × 权重 × 100")
    W = {"POR": 0.30, "PERM": 0.35, "SW": 0.35}
    for c in TARGETS:
        print(f"  {c:5s} 真实段占比 {is_real[c].mean():.4f} → acc_real +0.10 ≈ "
              f"{0.10 * is_real[c].mean() * W[c] * 100:+.3f} 分")
    print(f"\n用时 {time.time()-t0:.1f}s")
    print("口径：本脚本用**真值分段**（不可达），只回答「真实段回归本身还有没有余量」。")
    print("  要落地必须用预测分段 + 段内处理，收益会因误判边界而打折。")


if __name__ == "__main__":
    main()
