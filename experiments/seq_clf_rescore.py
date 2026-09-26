"""从已落盘的 pdf_va 重算全部下游分数（免重训）。

用途：`seq_clf_probe.py` 打分段要同时持有 v7 缓存 + 标签，本机内存紧张时可能在该段
被内核杀掉。而 pdf_va 在打分段之前就已经 `np.savez_compressed` 落盘了（seq_bilstm_pdf_<tag>.npz），
所以训练成果不会丢。本脚本只读那个 npz + v7 缓存，重算：
  基线自检 / drop-in 总分 / λ 融合 / 逐井 Δ / 留一井 jackknife / 分解。
口径与 seq_clf_probe.py 完全一致（v7 的 blend_w / clf_tcn_w，阈值只在验证集扫）。

用法: python experiments/seq_clf_rescore.py --tag seq_bilstm14
"""
from __future__ import annotations

import argparse
import glob
import gc
import os
import resource
import subprocess
import sys
import time
import warnings

warnings.filterwarnings("ignore")
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.feature import default_mask, make_targets  # noqa: E402
from src.utils import DEFAULT_VALUES, EPS, TARGETS, TRAIN_COLS, WEIGHTS  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "experiments", "runs")
N_TRAIN_WELLS = 72

V7 = {"blend_w": {"POR": 0.50, "PERM": 0.50, "SW": 0.75},
      "thresholds": {"POR": 0.130, "PERM": 0.270, "SW": 0.050},
      "clf_tcn_w": {"POR": 0.60, "PERM": 0.70, "SW": 0.30}}
V7_SCREEN = 73.749
TH_GRID = np.arange(0.02, 0.981, 0.01)
T0 = time.time()


def rss_gb():
    """峰值常驻集（GB）。macOS 的 ru_maxrss 单位随版本变：旧版 KB，Darwin 27 起是**字节**。

    用 `ps -o rss=`（恒定 KB）做参照判定单位：比值 ≥1000 说明 raw 是字节，否则是 KB。
    踩过一次：Darwin 27 上把字节当 KB 换算，把 2.62 GB 打成「2619.39 GB」。
    """
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    cur_kb = float(subprocess.check_output(
        ["ps", "-o", "rss=", "-p", str(os.getpid())]).strip() or 1.0)
    return raw / 1e9 if raw / cur_kb >= 1000.0 else raw / 1e6


def acc_vec(c, pred, true):
    p, t = np.asarray(pred, np.float64), np.asarray(true, np.float64)
    if c == "PERM":
        return np.maximum(0.0, 1.0 - np.abs(np.log10(np.maximum(p / np.maximum(t, EPS), EPS))))
    tol = 0.08 if c == "POR" else 0.05
    return np.maximum(0.0, 1.0 - np.abs(p - t) / (tol * (np.abs(t) + EPS)))


def sweep(pmv, regb, c, y, m):
    pmv, regb, y = pmv[m], regb[m], y[m]
    fd = acc_vec(c, np.full(len(pmv), DEFAULT_VALUES[c]), y)
    fr = acc_vec(c, regb, y)
    order = np.argsort(pmv, kind="stable")
    pos = np.searchsorted(pmv[order], TH_GRID, side="left")
    Fd = np.concatenate([[0.0], np.cumsum(fd[order])])
    Fr = np.concatenate([[0.0], np.cumsum(fr[order])])
    n = len(pmv)
    acc = (Fr[pos] + (Fd[n] - Fd[pos])) / n
    j = int(np.argmax(acc))
    return float(acc[j]), float(TH_GRID[j])


def score_at(pmv, regb, c, y, m, th):
    return float(acc_vec(c, np.where(pmv >= th, DEFAULT_VALUES[c], regb)[m], y[m]).mean())


def load_val_only():
    """只读 8 口验证井（不 load_wells 全 80 井，省数百 MB）。"""
    files = sorted(glob.glob(os.path.join(ROOT, "train", "*.txt")))[N_TRAIN_WELLS:]
    out = []
    for f in files:
        df = pd.read_csv(f, skiprows=2, names=TRAIN_COLS, dtype=float)
        df["logId"] = os.path.splitext(os.path.basename(f))[0]
        out.append(df)
    return pd.concat(out, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="seq_bilstm14")
    args = ap.parse_args()

    npz = np.load(os.path.join(RUNS, f"seq_bilstm_pdf_{args.tag}.npz"))
    pdf_va = np.float32(npz["pdf_va"])

    bnd = 657489  # 训练 72 井行数，与 prep() 一致
    Z = joblib.load(os.path.join(RUNS, "screen_tree_cache.joblib"))
    p_tree = {c: np.float32(Z["p_def"][c]) for c in TARGETS}
    r_tree = {c: np.float32(Z["reg"][c]) for c in TARGETS}
    T = np.load(os.path.join(RUNS, "tcn_all80_parts.npz"))
    reg_t = {c: np.float32(T[f"reg_{c}"][bnd:]) for c in TARGETS}
    pdf_t = {c: np.float32(T[f"pdf_{c}"][bnd:]) for c in TARGETS}
    del T, Z

    va = load_val_only()
    Y = make_targets(va)
    yva = {c: Y[c].to_numpy(np.float64) for c in TARGETS}
    valid = {c: ~np.isnan(yva[c]) for c in TARGETS}
    dv_mask = default_mask(Y)

    assert pdf_va.shape[0] == len(va) == 72779, (pdf_va.shape, len(va))
    assert len(p_tree["POR"]) == len(reg_t["POR"]) == len(va), "缓存长度不一致"
    print(f"载入 pdf_va {pdf_va.shape} / 验证 {len(va):,} 行   [{time.time()-T0:.0f}s]")

    # ── 基线自检：必须落在 73.749 ± 0.05 ──
    res_v7, tot_v7 = {}, 0.0
    for c in TARGETS:
        cw, bw = V7["clf_tcn_w"][c], V7["blend_w"][c]
        pm = (1 - cw) * p_tree[c] + cw * pdf_t[c]
        rt = r_tree[c].astype(np.float64)      # 缓存三目标全部已 log10，勿再取 log
        regb = np.power(10.0, (1 - bw) * rt + bw * reg_t[c])
        m = valid[c]
        a, th = sweep(pm, regb, c, yva[c], m)
        tot_v7 += WEIGHTS[c] * a * 100
        res_v7[c] = {"acc": a, "th": th}
    ok = abs(tot_v7 - V7_SCREEN) <= 0.05
    print(f"\n基线自检：harness v7 Total = {tot_v7:.3f}（部署 {V7_SCREEN}，±0.05）"
          f"→ {'PASS' if ok else 'FAIL'}   [{time.time()-T0:.0f}s]")

    # ── drop-in：BiLSTM 替换树的 p_def ──
    auc_seq, cand, tot_seq = {}, {}, 0.0
    for i, c in enumerate(TARGETS):
        cw, bw = V7["clf_tcn_w"][c], V7["blend_w"][c]
        pm = (1 - cw) * pdf_va[:, i].astype(np.float32) + cw * pdf_t[c]
        rt = r_tree[c].astype(np.float64)
        regb = np.power(10.0, (1 - bw) * rt + bw * reg_t[c])
        m = valid[c]
        a, th = sweep(pm, regb, c, yva[c], m)
        tot_seq += WEIGHTS[c] * a * 100
        cand[c] = {"acc": a, "th": th, "acc_v7th": score_at(pm, regb, c, yva[c], m, V7["thresholds"][c])}
        auc_seq[c] = float(roc_auc_score(dv_mask[c].to_numpy()[m], pdf_va[m, i]))

    print("\n官方 screen Total（BiLSTM 替换树 p_def；回归器/TCN/权重沿用 v7）")
    print(f"{'目标':6s}{'权重':>6s}{'v7 acc':>9s}{'v7 th':>7s}{'seq acc':>9s}{'seq th':>8s}"
          f"{'seq@v7th':>10s}{'raw AUC':>9s}{'Δ贡献':>9s}")
    for c in TARGETS:
        r0, r1 = res_v7[c], cand[c]
        print(f"{c:6s}{WEIGHTS[c]:6.2f}{r0['acc']:9.4f}{r0['th']:7.3f}"
              f"{r1['acc']:9.4f}{r1['th']:8.3f}{r1['acc_v7th']:10.4f}"
              f"{auc_seq[c]:9.4f}{WEIGHTS[c]*(r1['acc']-r0['acc'])*100:+9.3f}")
    dv = tot_seq - tot_v7
    print(f"\n  Total: v7 = {tot_v7:.3f}   BiLSTM = {tot_seq:.3f}   Δ = {dv:+.3f}"
          f"  → {'超出噪声带' if abs(dv) >= 0.3 else '在噪声带内'}（screen 噪声 ≈ ±0.3）")

    # ── λ 融合：BiLSTM 作第三路 ──
    print("\nλ 融合：pm = (1-λ)·pm_v7 + λ·p_BiLSTM")
    print(f"{'目标':6s}{'λ*':>7s}{'acc@λ*':>10s}{'acc@λ=0':>10s}{'最优th':>9s}{'Δ贡献':>10s}")
    dblend = 0.0
    for i, c in enumerate(TARGETS):
        cw, bw = V7["clf_tcn_w"][c], V7["blend_w"][c]
        pm0 = (1 - cw) * p_tree[c] + cw * pdf_t[c]
        rt = r_tree[c].astype(np.float64)
        regb = np.power(10.0, (1 - bw) * rt + bw * reg_t[c])
        m = valid[c]
        best_a, best_l, best_th = -1.0, 0.0, 0.0
        for lam in np.linspace(0.0, 1.0, 21):
            a, th = sweep(pm0 + (lam) * (pdf_va[:, i].astype(np.float32) - pm0),
                          regb, c, yva[c], m)
            if a > best_a:
                best_a, best_l, best_th = a, float(lam), th
        a0, _ = sweep(pm0, regb, c, yva[c], m)
        assert abs(a0 - res_v7[c]["acc"]) < 1e-9, f"{c}: λ=0 与自检不一致 {a0} vs {res_v7[c]['acc']}"
        d = (best_a - a0) * WEIGHTS[c] * 100
        dblend += d
        print(f"{c:6s}{best_l:>7.2f}{best_a:>10.4f}{a0:>10.4f}{best_th:>9.3f}{d:>+10.3f}")
    print(f"  Σ Δ贡献 = {dblend:+.3f}  → {'超出噪声带' if abs(dblend) >= 0.3 else '在噪声带内'}")

    # ── 逐井 Δ + 留一井 jackknife（按 logId 边界）──
    print("\n留一井 jackknife（逐井 Total：v7 vs BiLSTM，各自最优阈值用剩 7 口调）")
    print(f"{'井':32s}{'v7':>8s}{'seq':>8s}{'Δ':>9s}")
    bounds = []
    off = 0
    for _, g in va.groupby("logId", sort=False):
        off += len(g)
        bounds.append(off)
    assert bounds[-1] == len(va), (bounds[-1], len(va))

    loo, deltas = [], []
    for drop in range(len(bounds)):
        tot0 = tot1 = 0.0
        for i, c in enumerate(TARGETS):
            cw, bw = V7["clf_tcn_w"][c], V7["blend_w"][c]
            rt = r_tree[c].astype(np.float64)
            regb = np.power(10.0, (1 - bw) * rt + bw * reg_t[c])
            pm0 = (1 - cw) * p_tree[c] + cw * pdf_t[c]
            pm1 = (1 - cw) * pdf_va[:, i].astype(np.float32) + cw * pdf_t[c]
            m_all = valid[c]
            keep = np.ones(len(m_all), bool)
            lo = 0 if drop == 0 else bounds[drop - 1]
            keep[lo:bounds[drop]] = False
            mk = m_all & keep
            _, th0 = sweep(pm0, regb, c, yva[c], mk)
            _, th1 = sweep(pm1, regb, c, yva[c], mk)
            a0 = score_at(pm0, regb, c, yva[c], m_all, th0)
            a1 = score_at(pm1, regb, c, yva[c], m_all, th1)
            tot0 += WEIGHTS[c] * a0 * 100
            tot1 += WEIGHTS[c] * a1 * 100
        d = tot1 - tot0
        loo.append(d)
        deltas.append(d)
        print(f"drop_well{drop:02d}" + " " * 15 + f"{tot0:8.2f}{tot1:8.2f}{d:+9.3f}")
    loo = np.array(loo)
    np_pos = int((loo > 0).sum())
    print(f"\n  留一井 Δ ∈ [{loo.min():+.3f}, {loo.max():+.3f}]，"
          f"{np_pos}/8 为正，跨 0 = {'是' if loo.min() < 0 < loo.max() else '否'}")

    # ── 分解 def/real ──
    print("\n分解（各自最优阈值；def=真值默认段，real=真值真实段）")
    print(f"{'目标':6s}{'acc_def':>10s}{'acc_real':>11s}{'默认占比':>10s}")
    for i, c in enumerate(TARGETS):
        cw, bw = V7["clf_tcn_w"][c], V7["blend_w"][c]
        rt = r_tree[c].astype(np.float64)
        regb = np.power(10.0, (1 - bw) * rt + bw * reg_t[c])
        pm = (1 - cw) * pdf_va[:, i].astype(np.float32) + cw * pdf_t[c]
        m = valid[c]
        _, th = sweep(pm, regb, c, yva[c], m)
        pred = np.where(pm >= th, DEFAULT_VALUES[c], regb)
        dvm = dv_mask[c].to_numpy() & m
        ad = float(acc_vec(c, pred[m & dvm], yva[c][m & dvm]).mean()) if (m & dvm).sum() else float("nan")
        ar = float(acc_vec(c, pred[m & ~dvm], yva[c][m & ~dvm]).mean()) if (m & ~dvm).sum() else float("nan")
        print(f"{c:6s}{ad:10.4f}{ar:11.4f}{dvm.sum()/max(m.sum(),1):10.4f}")

    print(f"\n峰值 RSS = {rss_gb():.2f} GB   总耗时 = {time.time()-T0:.0f}s")

    out = {
        "baseline_selfcheck_total": round(tot_v7, 3),
        "selfcheck_ok": bool(ok),
        "screen_total": round(tot_seq, 3),
        "delta_vs_v7": round(dv, 3),
        "por_auc": round(auc_seq["POR"], 4),
        "perm_auc": round(auc_seq["PERM"], 4),
        "sw_auc": round(auc_seq["SW"], 4),
        "por_acc": round(cand["POR"]["acc"], 4),
        "perm_acc": round(cand["PERM"]["acc"], 4),
        "sw_acc": round(cand["SW"]["acc"], 4),
        "loo_min": round(float(loo.min()), 3),
        "loo_max": round(float(loo.max()), 3),
        "n_positive_wells": np_pos,
        "lambda_total_delta": round(dblend, 3),
        "max_rss_gb": round(rss_gb(), 2),
    }
    gc.collect()
    print("\n" + json_dumps(out))


def json_dumps(d):
    import json
    return json.dumps(d, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
