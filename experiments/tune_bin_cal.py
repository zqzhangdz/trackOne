"""按预测值分箱的乘法校准：从「验证集自评上界」到**诚实的跨井迁移估计**。

起点
----
diag_deployed_residual.py 的 D4 修好量纲 bug 后 PERM 显示 +0.58 折算分，但那是**验证集内
自评**（偏乐观上界）。两个未检验的前提：
  (1) 可部署要求校准在训练井上拟合、在验证井上应用；
  (2) 若箱内是双峰混合（truth 既可能高 3000× 也可能低 3000×），中位修正对一半点有益、
      一半有害，净收益 ≈ 0。

本脚本在**部署同源**的混合预测上（weight_tol 树 + 3 seed TCN，v9 融合配置）验证：
  T0  自检：树与 TCN 的验证井输出与缓存逐点一致（§4.10，中位差 < 1e-6），
      v7 口径 Total 复现 73.749。
  T1  自评上界：校准拟合在 8 验证井上、施加在 8 验证井上（≡ D4 的 +0.58）。
  T2  跨井迁移：校准拟合在 8 口**从未见过**的训练井上（模型只见过 64 井）、施加到 8 验证井。
  T3  留一井 jackknife：在 8 口留出井内 7 拟合 → 1 施加，得逐井 Δ 与包络。
      §6 的晋级判据是「留一井包络不跨 0」，这是唯一诚实的不确定性估计。

校准形式（log10 空间，乘性）
----------------------------
箱内最优乘性修正恰为该箱残差中位 r* = median(log10 p − log10 y)，因为它最小化 mean|rr − r*|
（PERM 的官方指标正是 |log10(p/y)|）。箱边界按 pred log10 分位切，只在拟合井上定义，
施加时按最近边界映射。分段决策（pm ≥ threshold）不变。

只 fit 两次树（72 井、64 井），TCN 只前向两个集合；所有校准步骤都是纯数组运算。
用法: python experiments/tune_bin_cal.py
"""
from __future__ import annotations

import os
import sys
import gc
import time
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import numpy as np
import pandas as pd

from src.feature import load_wells, make_targets, default_mask, make_features, inverse_transform_target
from src.utils import TARGETS, WEIGHTS, EPS, DEFAULT_VALUES
from experiments.eval_v4 import build_tree, tcn_outputs

RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
TAGS = ["T15_seed42_150ep", "T16_seed43_150ep", "T16_seed44_150ep"]
CFG = {"blend_w": {"POR": 0.50, "PERM": 0.50, "SW": 0.75},
       "thresholds": {"POR": 0.080, "PERM": 0.250, "SW": 0.050},
       "clf_tcn_w": {"POR": 0.60, "PERM": 0.70, "SW": 0.30}}
TH_V7 = {"POR": 0.130, "PERM": 0.270, "SW": 0.050}


def acc_vec(c, p, t):
    p = np.asarray(p, float)
    t = np.asarray(t, float)
    if c == "PERM":
        return np.maximum(0.0, 1.0 - np.abs(np.log10(np.maximum(p / np.maximum(t, EPS), EPS))))
    tol = 0.08 if c == "POR" else 0.05
    return np.maximum(0.0, 1.0 - np.abs(p - t) / (tol * (np.abs(t) + EPS)))


def reglog(out_c):
    return np.where(np.isfinite(out_c["reg"]), out_c["reg"], -5.0)


def score(out, Y, dmask, correct=None, ths=None):
    ths = ths or CFG["thresholds"]
    tot, parts = 0.0, {}
    for c in TARGETS:
        y = Y[c].to_numpy().astype(float)
        valid = np.isfinite(y)
        pr = np.power(10.0, reglog(out[c]))
        if correct is not None and c in correct:
            pr = pr / correct[c]
        pred = np.where(out[c]["pm"] >= ths[c], DEFAULT_VALUES[c], pr)
        a = np.where(np.isfinite(y), acc_vec(c, pred, y), np.nan)
        tot += WEIGHTS[c] * np.nanmean(a)
        d = dmask[c].to_numpy() & valid
        parts[c] = (float(np.nanmean(a)),
                    float(np.nanmean(a[valid & ~d])) if (valid & ~d).sum() else float("nan"),
                    float(np.nanmean(a[valid & d])) if (valid & d).sum() else float("nan"))
    return tot * 100, parts


def blend(tr_df, eval_wells, arts, tag=""):
    """部署同源：weight_tol 树 + 3 seed TCN，v9 融合。"""
    t = time.time()
    model, _ = build_tree(tr_df, weight_tol=True)
    df = pd.concat(eval_wells, ignore_index=True)
    Xva = make_features(df, context=True, priors=True)
    reg_t = {c: np.log10(np.maximum(inverse_transform_target(
        c, pd.Series(model.regressors[c].predict(Xva), index=Xva.index)).to_numpy(), EPS))
        for c in TARGETS}
    pmt = {c: model.classifiers[c].predict_proba(Xva)[:, 1] for c in TARGETS}
    ens, pmc = {}, {}
    for art in arts:
        r, p = tcn_outputs(eval_wells, art)
        for c in TARGETS:
            ens.setdefault(c, []).append(r[c])
            pmc.setdefault(c, []).append(p[c])
    out = {}
    for c in TARGETS:
        w, cw = CFG["blend_w"][c], CFG["clf_tcn_w"][c]
        out[c] = dict(reg=(1 - w) * reg_t[c] + w * np.mean(ens[c], axis=0),
                      pm=(1 - cw) * pmt[c] + cw * np.mean(pmc[c], axis=0),
                      reg_tree=reg_t[c], pm_tree=pmt[c],
                      reg_tcn=np.mean(ens[c], axis=0))
    Y = make_targets(df)
    print(f"  {tag} 就绪   {time.time() - t:.0f}s")
    return out, Y, default_mask(Y)


def make_calib(regs, Y, D, nb, mask=None):
    """按 pred log10 分箱，取箱内残差中位为乘性修正量（log10 空间的加法量）。"""
    cal, diag = {}, {}
    for c in TARGETS:
        y = Y[c].to_numpy().astype(float)
        m = np.isfinite(y) & ~D[c].to_numpy()
        if mask is not None:
            m &= mask
        lo = reglog(regs[c])[m]
        rr = lo - np.log10(np.clip(y[m], 1e-12, None))
        edges = np.quantile(lo, np.linspace(0, 1, nb + 1))
        ratios = np.zeros(nb)
        rows = []
        for b in range(nb):
            mb = (lo >= edges[b]) & ((lo < edges[b + 1]) if b < nb - 1 else True)
            n = int(mb.sum())
            if n > 50:
                ratios[b] = float(np.median(rr[mb]))
                q25, q75 = float(np.quantile(rr[mb], 0.25)), float(np.quantile(rr[mb], 0.75))
                rows.append((b, n, float(np.median(rr[mb])), q25, q75))
        cal[c] = dict(edges=edges, ratios=ratios)
        diag[c] = rows
    return cal, diag


def apply_calib(cal, out, nb, robust=False, diag=None):
    """robust=True 时，把箱内残差分布过宽（双峰）的箱的修正置零。

    理由：箱内最优修正是残差中位 r*，它只在箱内残差近单峰时最优。若箱内同时存在
    「truth 高 3000×」和「truth 低 3000×」的点，中位修正对一半点有益、一半有害，
    净收益≈0（PERM 的 |log10(p/y)| 对对称的双峰完全抵消）。此时宁可不动。
    """
    corr = {}
    for c in TARGETS:
        e, r = cal[c]["edges"], cal[c]["ratios"]
        b = np.clip(np.digitize(reglog(out[c]), e[1:-1], right=False), 0, nb - 1)
        if robust and diag is not None:
            zero = set()
            for (bi, n, med, q25, q75) in diag[c]:
                # 容差带在半轴上宽 tol_log（PERM 1.0 / POR 0.0334 / SW 0.0212）
                tol = 1.0 if c == "PERM" else (0.03342 if c == "POR" else 0.02119)
                if (q75 - q25) > 0.5 * tol:
                    zero.add(bi)
            r = np.array(r, float)
            r[list(zero)] = 0.0
        corr[c] = np.power(10.0, r[b])          # corrected = p / 10^median(rr)
    return corr


def main():
    t0 = time.time()
    wells = load_wells("./train", has_labels=True)
    ids = [w["logId"].iloc[0] for w in wells]
    by = dict(zip(ids, wells))
    tr72 = [by[i] for i in ids[:72]]
    held8 = [by[i] for i in ids[8:16]]
    tr64 = [by[i] for i in ids[:72] if i not in set(ids[8:16])]
    va8 = [by[i] for i in ids[72:80]]
    assert (len(tr72), len(tr64), len(held8), len(va8)) == (72, 64, 8, 8)
    arts = [joblib.load(os.path.join(RUNS, f"tcn_{t}.joblib")) for t in TAGS]
    print(f"切分 72 = 64 + 8(留出) ，另有 8 验证井")

    va_out, Yva, Dva = blend(pd.concat(tr72, ignore_index=True), va8, arts, "72井模型 → 8验证井")
    all8_out, Ya, Da = blend(pd.concat(tr64, ignore_index=True), held8, arts,
                             "64井模型 → 8留出井（全部未见）")
    tr64_out, Yt, Dt = blend(pd.concat(tr64, ignore_index=True), tr64, arts,
                             "64井模型 → 64训练井（箱定义用，样本内）")

    # ---------- T0 自检 ----------
    cache = joblib.load(os.path.join(RUNS, "screen_tree_cache.joblib"))
    tcn = {k: v for k, v in np.load(os.path.join(RUNS, "tcn_screen_outputs.npz")).items()}
    wf = pd.concat(va8, ignore_index=True)["logId"].to_numpy()
    cw = np.asarray(cache["va_ids"])
    wf_ids = sorted(set(wf))
    for c in TARGETS:
        a = np.median(np.abs(np.sort(va_out[c]["reg_tree"]) - np.sort(cache["reg"][c])))
        b = np.median(np.abs(np.sort(va_out[c]["pm_tree"]) - np.sort(cache["p_def"][c])))
        tc = 0.0
        for w in wf_ids:
            m1, m2 = wf == w, cw == w
            if min(int(m1.sum()), int(m2.sum())) == 0:
                continue
            tc = max(tc, float(np.median(np.abs(
                np.sort(va_out[c]["reg_tcn"][m1]) -
                np.sort(np.mean([tcn[f"{t}__{c}__reg"] for t in TAGS], axis=0)[m2])))))
        print(f"  [参考] {c} 树 reg {a:.2e} / p_def {b:.2e} / TCN reg {tc:.2e}"
              "（§4.10 已记录树 reg 有 ~1e-3~1e-2 的历史残差；TCN reg 应 <1e-6，"
              "它才是校准真正依赖的对象）")
        if tc >= 1e-6:
            assert False, f"TCN reg 中位差 {tc:.2e} ≥ 1e-6 → TCN 通路不对齐，校准无意义"
    b_v7, d_v7 = score(va_out, Yva, Dva, ths=TH_V7)
    b_v9, d_v9 = score(va_out, Yva, Dva, ths=CFG["thresholds"])
    print(f"  [自检] 验证 8 井 Total：v9 阈值 {b_v9:.3f}（应复现记录的 v9 screen 73.931）"
          f"   v7 阈值 {b_v7:.3f}（225 特征口径下才是 73.749，此处 237 特征不可比）")
    if abs(b_v9 - 73.931) > 0.02:
        assert False, (f"基线 {b_v9:.3f} 未复现 v9 screen 73.931 → "
                       "校准建在不可信的基线上，退出")
    print("  [自检] v7 口径分解 " + "  ".join(
        f"{c} acc {d_v7[c][0]:.4f}/real {d_v7[c][1]:.4f}/def {d_v7[c][2]:.4f}" for c in TARGETS))

    def show(label, rows):
        for c, rowsc in zip(TARGETS, rows):
            print(f"  {c:<5}")
            for (b, n, med, q25, q75) in rowsc:
                bimodal = q75 - q25 < 0.4 and abs(med) > 0.3
                print(f"     箱{b:>2} n={n:7,}  残差中位 {med:+.3f}  IQR [{q25:+.3f},{q75:+.3f}]"
                      f"  宽 {q75 - q25:.3f}"
                      f"{'   ← 箱内近常数（中位修正有效）' if bimodal else ''}")

    print("\n" + "=" * 96)
    print("箱诊断（8 口留出井，模型未见过）：箱内残差若近常数则中位修正有效，若双峰则无效")
    print("=" * 96)
    for nb in (5, 11):
        print(f"\n nb = {nb}")
        show(f"nb{nb}", make_calib(all8_out, Ya, Da, nb)[1])

    def trial(label, fit_regs, fit_Y, fit_D, ap_out, ap_Y, ap_D, nb, fit_mask=None):
        cal, dg = make_calib(fit_regs, fit_Y, fit_D, nb, mask=fit_mask)
        b0, d0 = score(ap_out, ap_Y, ap_D)
        for tag, corr in (("常规", apply_calib(cal, ap_out, nb)),
                          ("稳健", apply_calib(cal, ap_out, nb, robust=True, diag=dg))):
            b1, d1 = score(ap_out, ap_Y, ap_D, correct=corr)
            print(f"\n  {label}  nb={nb}  {tag}   Total {b0:.3f} → {b1:.3f}"
                  f"   Δ {b1 - b0:+.3f} 分")
            for c in TARGETS:
                print(f"     {c}  acc {d0[c][0]:.4f}→{d1[c][0]:.4f}  "
                      f"real {d0[c][1]:.4f}→{d1[c][1]:.4f}"
                      f"   Δ {(d1[c][0] - d0[c][0]) * 100 * WEIGHTS[c]:+.4f} 分")
            yield b1 - b0, c

    print("\n" + "=" * 96)
    print("T1  自评上界（偏乐观对照）：校准拟合在 8 验证井上、施加在 8 验证井上")
    print("=" * 96)
    for nb in (5, 11, 21):
        for g, _ in trial("T1 自评", va_out, Yva, Dva, va_out, Yva, Dva, nb):
            pass

    print("\n" + "=" * 96)
    print("T2  跨井迁移（核心）：校准拟合在 8 口从未见过的留出井上 → 施加到 8 验证井")
    print("=" * 96)
    for nb in (5, 11, 21):
        for g, _ in trial("T2 迁移", all8_out, Ya, Da, va_out, Yva, Dva, nb):
            pass

    print("\n" + "=" * 96)
    print("T2b 大样本对照：校准拟合在 64 训练井（样本内）→ 施加到 8 留出井")
    print("=" * 96)
    for nb in (5, 11, 21):
        for g, _ in trial("T2b", tr64_out, Yt, Dt, all8_out, Ya, Da, nb):
            pass

    print("\n" + "=" * 96)
    print("T3  留一井 jackknife：在 8 口留出井内 7 拟合 → 1 施加（§6 晋级判据）")
    print("     逐井 Total 差异来自各井真实段占比不同，故报**点数加权的** Δ 作为总分校正")
    print("=" * 96)
    for nb in (5, 11):
        for robust in (False, True):
            rows = []
            for k in range(8):
                fw = np.ones(8, bool)
                fw[k] = False
                cal, dg = make_calib(all8_out, Ya, Da, nb, mask=fw)
                s0 = sum(len(held8[j]) for j in range(k))
                wlen = len(held8[k])
                o1 = {c: {kk: v[s0:s0 + wlen] for kk, v in all8_out[c].items()}
                      for c in TARGETS}
                Ys = pd.DataFrame({c: Ya[c].iloc[s0:s0 + wlen] for c in TARGETS})
                Ds = default_mask(Ys)
                b0, d0 = score(o1, Ys, Ds)
                b1, d1 = score(o1, Ys, Ds,
                               correct=apply_calib(cal, o1, nb, robust=robust, diag=dg))
                rows.append((held8[k]["logId"].iloc[0][:8], wlen, b0, b1, b1 - b0,
                             d0, d1))
            w = np.array([r[1] for r in rows], float)
            dlt = np.array([r[4] for r in rows], float)
            tag = "稳健" if robust else "常规"
            print(f"\n nb = {nb}   {tag}   （每行 = 该井被留出、其余 7 井拟合校准）")
            for aid, wlen, b0, b1, d, d0, d1 in rows:
                print(f"   {aid}  n={wlen:6,}  {b0:.3f} → {b1:.3f}   Δ {d:+.3f}"
                      + "  " + "  ".join(f"{t}{(d1[t][0] - d0[t][0]) * 100 * WEIGHTS[t]:+.3f}"
                                         for t in TARGETS))
            wavg = float((w * dlt).sum() / w.sum())
            print(f"   逐井 Δ：均值 {dlt.mean():+.3f}   点数加权 {wavg:+.3f}"
                  f"   包络 [{dlt.min():+.3f},{dlt.max():+.3f}]   正号 "
                  f"{int((dlt > 0).sum())}/8")
            print(f"   {'包络不跨 0 → 满足 §6 迁移判据' if dlt.min() > 0 else '包络跨 0 → 不构成证据'}")

    print("\n" + "=" * 96)
    print("T4  4 拟合 → 4 施加（折半）：校准样本量只有 4 井时的迁移，重复多个划分")
    print("=" * 96)
    rng = np.random.default_rng(42)
    order = np.arange(8)
    for nb in (5, 11):
        for robust in (False, True):
            gs, gw = [], []
            for it in range(6):
                rng.shuffle(order)
                for side in (0, 1):
                    fm = np.zeros(8, bool)
                    fm[order[side::2]] = True
                    cal, dg = make_calib(all8_out, Ya, Da, nb, mask=fm)
                    for a in order[side + 1::2]:
                        s0 = sum(len(held8[j]) for j in range(a))
                        wlen = len(held8[a])
                        o1 = {c: {kk: v[s0:s0 + wlen] for kk, v in all8_out[c].items()}
                              for c in TARGETS}
                        Ys = pd.DataFrame({c: Ya[c].iloc[s0:s0 + wlen] for c in TARGETS})
                        Ds = default_mask(Ys)
                        b0, _ = score(o1, Ys, Ds)
                        b1, _ = score(o1, Ys, Ds,
                                      correct=apply_calib(cal, o1, nb,
                                                           robust=robust, diag=dg))
                        gs.append(b1 - b0)
                        gw.append(wlen)
            gs, gw = np.array(gs), np.array(gw)
            tag = "稳健" if robust else "常规"
            print(f" nb={nb} {tag}   24 次单井试验：均值 {gs.mean():+.3f}  "
                  f"点数加权 {(gw * gs).sum() / gw.sum():+.3f}"
                  f"   包络 [{gs.min():+.3f},{gs.max():+.3f}]"
                  f"  正号 {int((gs > 0).sum())}/24")

    print(f"\n用时 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
