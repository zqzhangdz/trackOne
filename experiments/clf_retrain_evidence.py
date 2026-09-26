"""v8 候选证据复核：逐井分解 + 留一井 jackknife（纯 sklearn，不跑 TCN）。

背景：`tune_clf_metric.py` 的 15 个候选已跑完（结果见 `experiments/runs/clf_retrain.log`，
两轮独立运行结果完全一致）。每目标取最优后，相对 v7 基线在 screen 上只值 +0.055～+0.090 分，
落在 §6 的 ±0.3 噪声带内。

关键的方法学问题：72,779 个验证行被 8 口井高度聚类（单井内 POR lag-1 自相关 0.967），
行级 bootstrap 的 SE 与 1/√72779 成正比，会低估方差一个数量级。8 口井才是实验单元，
所以这里只按井做证据：留一井 jackknife。

每目标最优候选（来自 clf_retrain.log）：
  POR  → mw       (判错代价加权, th 0.230)
  PERM → mw       (判错代价加权, th 0.320)
  SW   → cross+reg (另两目标 p_def + 本目标 TCN reg/pdf, th 0.070)

bal 直接复用缓存的 p_val，只需重训 3 个非基线分类器，约 90 秒。
用法: python experiments/clf_retrain_evidence.py
"""
from __future__ import annotations
import gc, os, sys, time, warnings
warnings.filterwarnings("ignore")
import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sklearn.ensemble import HistGradientBoostingClassifier
from src.feature import default_mask, load_wells, make_features, make_targets
from src.utils import DEFAULT_VALUES, EPS, TARGETS, WEIGHTS

RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
BND = 657_489
V7 = {"blend_w": {"POR": 0.50, "PERM": 0.50, "SW": 0.75},
      "thresholds": {"POR": 0.130, "PERM": 0.270, "SW": 0.050},
      "clf_tcn_w": {"POR": 0.60, "PERM": 0.70, "SW": 0.30}}
# 每目标最优候选：(候选名, 最优阈值, 特征集, 样本加权模式)
PICK = {"POR": ("mw", 0.230, "base", "regret"),
        "PERM": ("mw", 0.320, "base", "regret"),
        "SW": ("cross+reg", 0.070, "crossreg", None)}
T0 = time.time()


def acc_vec(c, pred, true):
    p = np.asarray(pred, np.float64); t = np.asarray(true, np.float64)
    if c == "PERM":
        return np.maximum(0.0, 1.0 - np.abs(np.log10(np.maximum(p / np.maximum(t, EPS), EPS))))
    tol = 0.08 if c == "POR" else 0.05
    return np.maximum(0.0, 1.0 - np.abs(p - t) / (tol * (np.abs(t) + EPS)))


def regret_weight(c, y, reg_phys):
    """逐点「判错的代价」= |acc(判默认) − acc(判真实)|，与 tune_clf_metric.py 一致。"""
    a_d = acc_vec(c, np.full(len(y), DEFAULT_VALUES[c]), y)
    a_r = acc_vec(c, reg_phys, y)
    w = np.abs(a_d - a_r)
    w = np.clip(w, np.percentile(w, 5), np.percentile(w, 95))
    w = np.maximum(w, 1e-3)
    return (w / w.mean()).astype(np.float32)


def main() -> None:
    print("载入 80 井 ...", flush=True)
    wells = load_wells("./train", has_labels=True)
    df = pd.concat(wells, ignore_index=True)
    row_wells = df["logId"].to_numpy()
    X = make_features(df, context=True, priors=True)
    Y = make_targets(df)
    DM = default_mask(Y)
    del wells, df; gc.collect()
    X_tr, X_va = X.iloc[:BND], X.iloc[BND:]
    Y_tr, Y_va = Y.iloc[:BND], Y.iloc[BND:]
    D_tr, D_va = DM.iloc[:BND], DM.iloc[BND:]
    y_all = {c: Y[c].to_numpy().astype(np.float64) for c in TARGETS}
    del X, Y, DM; gc.collect()
    print(f"  训练 {BND:,} × {X_tr.shape[1]} / 验证 {len(X_va):,}  [{time.time()-T0:.0f}s]", flush=True)

    d = np.load(os.path.join(RUNS, "tcn_all80_parts.npz"))
    ens = {c: np.float32(d[f"reg_{c}"]) for c in TARGETS}
    pdf = {c: np.float32(d[f"pdf_{c}"]) for c in TARGETS}
    ck = joblib.load(os.path.join(RUNS, "clf_retrain_inputs.joblib"))
    rt_all, p_val, oof = ck["rt_all"], ck["p_val"], ck["oof"]
    del ck; gc.collect()

    y_tr = {c: y_all[c][:BND] for c in TARGETS}
    y_va = {c: y_all[c][BND:] for c in TARGETS}
    m_tr = {c: ~np.isnan(y_tr[c]) for c in TARGETS}
    m_va = {c: ~np.isnan(y_va[c]) for c in TARGETS}

    # ── 逐目标：基线 bal 与最优候选的 p_def（验证侧）
    p_def = {}
    for c in TARGETS:
        name, th, fset, swmode = PICK[c]
        others = [o for o in TARGETS if o != c]
        Xt = X_tr.to_numpy(np.float32); Xv = X_va.to_numpy(np.float32)
        if fset == "base":
            Xt_f, Xv_f = Xt, Xv
        elif fset == "crossreg":
            Xt_f = np.concatenate([Xt, oof[others[0]][:, None], oof[others[1]][:, None],
                                   ens[c][:BND][:, None], pdf[c][:BND][:, None]], axis=1)
            Xv_f = np.concatenate([Xv, p_val[others[0]][:, None], p_val[others[1]][:, None],
                                   ens[c][BND:][:, None], pdf[c][BND:][:, None]], axis=1)
        clf = HistGradientBoostingClassifier(
            random_state=42, learning_rate=0.1, max_iter=300, max_leaf_nodes=31,
            min_samples_leaf=50, l2_regularization=1.0, class_weight=None)
        sw = None
        if swmode:
            blend = (1 - V7["blend_w"][c]) * rt_all[c][:BND] + V7["blend_w"][c] * ens[c][:BND]
            sw = regret_weight(c, y_tr[c][m_tr[c]], np.power(10.0, blend)[m_tr[c]])
        clf.fit(Xt_f[m_tr[c]], D_tr[c].to_numpy().astype(int)[m_tr[c]].astype(np.int32),
                sample_weight=sw)
        p = clf.predict_proba(Xv_f)[:, 1].astype(np.float32)
        p_def[c] = (p_val[c].astype(np.float32), p, th)   # (bal, 候选, 候选最优阈值)
        del clf, Xt, Xv, Xt_f, Xv_f, sw; gc.collect()
        print(f"  {c} 候选 {name} 重训完成  [{time.time()-T0:.0f}s]", flush=True)

    # ── 逐行对总分的贡献
    vw = row_wells[BND:]
    lids = pd.unique(vw)
    regb = {c: np.power(10.0, (1 - V7["blend_w"][c]) * rt_all[c][BND:]
                        + V7["blend_w"][c] * ens[c][BND:]) for c in TARGETS}
    rowd = {c: np.zeros(len(vw), np.float64) for c in TARGETS}
    tot_bal = tot_best = 0.0
    print("\n" + "=" * 100)
    print("每目标 acc（bal = v7 部署配置；候选 = 该目标最优候选 + 其最优阈值）")
    print("=" * 100)
    print(f"{'目标':6s} {'权重':>6s} {'bal acc':>9s} {'候选 acc':>9s} {'Δacc':>8s} {'Δ贡献分':>9s}")
    for c in TARGETS:
        bal, cand, th = p_def[c]
        m = m_va[c]
        pb = (1 - V7["clf_tcn_w"][c]) * bal + V7["clf_tcn_w"][c] * pdf[c][BND:]
        pc = (1 - V7["clf_tcn_w"][c]) * cand + V7["clf_tcn_w"][c] * pdf[c][BND:]
        pred_b = np.where(pb >= V7["thresholds"][c], DEFAULT_VALUES[c], regb[c])
        pred_c = np.where(pc >= th, DEFAULT_VALUES[c], regb[c])
        a_b = float(acc_vec(c, pred_b[m], y_va[c][m]).mean())
        a_c = float(acc_vec(c, pred_c[m], y_va[c][m]).mean())
        contrib = WEIGHTS[c] * a_c * 100
        tot_bal += WEIGHTS[c] * a_b * 100
        tot_best += contrib
        # 逐行贡献：对总分的增量（含权重与有效点归一化）
        idx = np.flatnonzero(m)
        rowd[c][idx] = WEIGHTS[c] * (acc_vec(c, pred_c[m], y_va[c][m])
                                     - acc_vec(c, pred_b[m], y_va[c][m])) / int(m.sum())
        print(f"{c:6s} {WEIGHTS[c]:6.2f} {a_b:9.4f} {a_c:9.4f} {a_c-a_b:+8.4f} "
              f"{(a_c-a_b)*WEIGHTS[c]*100:+9.3f}   (阈值 {V7['thresholds'][c]:.3f} → {th:.3f})")
    print("-" * 100)
    print(f"{'Total':6s} {'':>6s} {'':>9s} {'':>9s} {'':>8s} {tot_best-tot_bal:+9.3f}")

    # ── 逐井分解 + 留一井 jackknife（8 井才是实验单元）
    print("\n" + "=" * 100)
    print("逐井分解 Δ 总分（v7 部署 vs 每目标最优候选）")
    print("=" * 100)
    print(f"{'井':34s} {'Δ总分':>8s}")
    wdelta = {}
    for lid in lids:
        sel = vw == lid
        dv = float(sum(rowd[c] for c in TARGETS)[sel].sum()) * 100
        wdelta[lid] = dv
        print(f"{lid:34s} {dv:+8.3f}")
    total_d = sum(wdelta.values())
    pos = sum(1 for v in wdelta.values() if v > 0)
    lo = total_d - max(wdelta.values())   # 去掉最大正贡献井
    hi = total_d - min(wdelta.values())   # 去掉最大负贡献井
    print("-" * 100)
    print(f"  合计 Δ = {total_d:+.3f}    v7 = {tot_bal:.3f}  →  候选 = {tot_best:.3f}")
    print(f"  正号井 {pos}/{len(lids)}")
    print(f"  留一井 jackknife [{lo:+.3f}, {hi:+.3f}]"
          f"  → {'跨 0：噪声内，不构成晋级证据' if lo <= 0 <= hi else '不跨 0：方向可信'}")
    print(f"  screen 噪声 ≈ ±0.3；本次 |Δ| = {abs(total_d):.3f} "
          f"{'在噪声带内' if abs(total_d) < 0.3 else '超出噪声带'}")
    print(f"\n  耗时 {time.time()-T0:.0f}s")


if __name__ == "__main__":
    main()
