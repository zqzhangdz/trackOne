"""井级残差校正器：一阶段的系统性偏差能否按「井」预测并修正？

为什么这件事值得单独验证
------------------------
diag_deployed_residual.py（部署同源混合预测，验证 8 井）显示真实段残差 log10 的
  POR   跨井 std 0.8805 / 井内 std 0.3998  极差 +2.693
  PERM  跨井 std 0.9741 / 井内 std 0.4502  极差 +3.038
  SW    跨井 std 0.0177 / 井内 std 0.1394  极差 +0.052   ← SW 是点级噪声
即 POR / PERM 的误差主要是**井级系统性乘性偏差**，且 logId 在推理时已知。若每口井的残差
可由井级统计量（均值/分位/偏度 + 井内曲线相关）预测，那就是此前从未试过的可部署杠杆：
  - 不是「平滑」（§7，按深度移动预测）——这里是按井平移；
  - 不是「逐井 pct-rank」（§7）——那不是对残差建模；
  - 不是「逐井 NN 迁移」（§7）——那要用近邻井的标签，这里只用本井曲线统计量；
  - 不是「per-well 阈值校准」（tune_transductive_cal，Δ 上界 +0.261）——那是分段阈值。

口径（避免泄漏，两处必须做对）
------------------------------
1) 一阶段 OOF 必须**按井分折**（GroupKFold）。diag_second_stage.py 的随机 5 折把同一口井
   分到不同折，逐井残差中位被平均掉，跨井 std 被压到采样噪声量级（POR 0.0061），
   会得出「无井级偏差」的**假阴性**。
2) 官方容差折算到 log10 空间不是 0.08/1.0/0.05，而是 |log10(ŷ/y)| ≤ log10(1+tol)：
   POR 0.03342 / PERM 1.0 / SW 0.02119。PERM 的 1.0 decade 恰好等于其比值容差，巧合。

用法: python experiments/diag_welllevel_corrector.py
"""
from __future__ import annotations

import os
import sys
import gc
import time
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

from src.feature import load_wells, make_targets, default_mask, make_features
from src.utils import TARGETS, WEIGHTS, EPS

RAW = ["DEPTH", "GR", "PE", "SP", "CAL", "AC", "DEN", "CNL", "RXO", "RT", "DEVI", "AZIM", "BIT"]
TOL = {"POR": 0.08, "PERM": 1.0, "SW": 0.05}
# 容差折算到 log10 空间：|log10(ŷ/y)| ≤ log10(1 + tol)
TOL_LOG = {c: (1.0 if c == "PERM" else float(np.log10(1.0 + TOL[c]))) for c in TARGETS}


def rss() -> str:
    import resource
    return f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576:.0f}MB"


def acc_vec(c: str, p, t):
    p = np.asarray(p, float)
    t = np.asarray(t, float)
    if c == "PERM":
        return np.maximum(0.0, 1.0 - np.abs(np.log10(np.maximum(p / np.maximum(t, EPS), EPS))))
    return np.maximum(0.0, 1.0 - np.abs(p - t) / (TOL[c] * (np.abs(t) + EPS)))


def logify(arr):
    return np.log10(np.clip(np.asarray(arr, float), 1e-12, None))


def well_stats(df: pd.DataFrame, wels: list[str]) -> np.ndarray:
    """井级统计量（推理时无标签可算）：每口井一行。"""
    per_curve = []
    for c in RAW:
        rows = []
        for w in wels:
            s = df.loc[df["logId"] == w, c].astype(float).to_numpy()
            s = s[np.isfinite(s)]
            if s.size < 10:
                rows.append(np.array([np.nan] * 11))
            else:
                rows.append([s.mean(), s.std(ddof=0), np.median(s), s.min(), s.max(),
                             np.percentile(s, 25), np.percentile(s, 75),
                             np.percentile(s, 5), np.percentile(s, 95),
                             pd.Series(s).skew(), pd.Series(s).kurt()])
        per_curve.append(np.array(rows, dtype=np.float64))       # n_well x 13 x 11
    W = [np.stack(per_curve, axis=1).reshape(len(wels), -1)]
    # 井内两两曲线相关（13 条曲线 → 78 个常数）——必须在**深度序列**上算，不是统计量上
    corr = []
    for w in wels:
        sub = df.loc[df["logId"] == w]
        v = []
        for i in range(len(RAW)):
            a = sub[RAW[i]].astype(float).to_numpy()
            for j in range(i + 1, len(RAW)):
                b = sub[RAW[j]].astype(float).to_numpy()
                aa = a[np.isfinite(a)]
                bb = b[np.isfinite(b)]
                if min(aa.size, bb.size) < 10 or np.ptp(aa) < 1e-12 or np.ptp(bb) < 1e-12:
                    v.append(0.0)
                else:
                    v.append(float(np.corrcoef(aa, bb)[0, 1]))
        corr.append(v)
    W.append(np.array(corr, dtype=np.float64))
    S = np.concatenate(W, axis=1)
    return np.nan_to_num(S, nan=0.0, posinf=0.0, neginf=0.0)


def hgb():
    return HistGradientBoostingRegressor(
        max_iter=250, learning_rate=0.06, max_leaf_nodes=63, min_samples_leaf=40,
        l2_regularization=1.0, loss="squared_error", random_state=42)


def main():
    t0 = time.time()
    wells = load_wells("./train", has_labels=True)
    ids = [w["logId"].iloc[0] for w in wells]
    tr = pd.concat([w for w in wells if w["logId"].iloc[0] in set(ids[:72])], ignore_index=True)
    va = pd.concat([w for w in wells if w["logId"].iloc[0] in set(ids[72:])], ignore_index=True)
    del wells
    gc.collect()
    tr = tr.sort_index().reset_index(drop=True)
    va = va.sort_index().reset_index(drop=True)
    ytr, yva = make_targets(tr), make_targets(va)
    dtr, dva = default_mask(ytr), default_mask(yva)
    Xtr = make_features(tr, context=True, priors=True).to_numpy(np.float32)
    Xva = make_features(va, context=True, priors=True).to_numpy(np.float32)
    wtr = tr["logId"].to_numpy()
    wva = va["logId"].to_numpy()
    uwtr = list(pd.unique(wtr))
    uwva = list(pd.unique(wva))
    S_tr, S_va = well_stats(tr, uwtr), well_stats(va, uwva)
    print(f"训练 {len(tr):,} 点 / {len(uwtr)} 井   验证 {len(va):,} 点 / {len(uwva)} 井")
    print(f"特征 {Xtr.shape[1]}   井级统计量 {S_tr.shape[1]}   {rss()}")
    print(f"log10 空间容差 {[round(TOL_LOG[c], 5) for c in TARGETS]}")

    gkf = GroupKFold(n_splits=5)
    fold = np.full(len(tr), -1, dtype=int)
    for fi, (_, oi) in enumerate(gkf.split(Xtr, groups=wtr)):
        fold[oi] = fi
    # 自检：折内不得出现同一口井（按井分折是本脚本成立的前提）
    bad = 0
    for fi in range(5):
        oi = fold == fi
        ii = np.ones(len(tr), bool)
        ii[oi] = False
        if len(set(wtr[oi]) & set(wtr[ii])) > 0:
            bad += 1
    assert bad == 0, "GroupKFold 折间出现同一口井，OOF 残差泄漏"
    print("[自检] GroupKFold 折间无同一口井 ✓")

    for c in TARGETS:
        yr = logify(ytr[c].to_numpy(float))
        yv = logify(yva[c].to_numpy(float))
        rt_ = np.isfinite(yr) & np.isfinite(Xtr).all(axis=1) & ~dtr[c].to_numpy()
        rv_ = np.isfinite(yv) & np.isfinite(Xva).all(axis=1) & ~dva[c].to_numpy()
        print(f"\n{'=' * 96}\n目标 {c}   真实段 训练 {rt_.sum():,} / 验证 {rv_.sum():,}")

        # ---- 一阶段：按井 5 折 OOF（训练井）+ 全量 fit（验证井）----
        p_oof = np.full(len(tr), np.nan)
        for fi in range(5):
            oi = fold == fi
            ii = ~oi
            m = hgb()
            m.fit(Xtr[rt_ & ii], yr[rt_ & ii])
            p_oof[oi] = m.predict(Xtr[oi])
            del m
            gc.collect()
        m_full = hgb()
        m_full.fit(Xtr[rt_], yr[rt_])
        p_va = m_full.predict(Xva[rv_])
        tv = yv[rv_]
        base = acc_vec(c, np.power(10.0, p_va), np.power(10.0, tv)).mean()
        rr_va = p_va - tv
        print(f"  一阶段验证集 真实段 acc {base:.4f}   log10-MAE "
              f"{np.nanmean(np.abs(rr_va)):.4f}")

        # ---- A 训练井逐井残差（按井 OOF，无泄漏）----
        ro = p_oof - yr
        per_well_r, valid_w, Sw = [], [], []
        for k, w in enumerate(uwtr):
            mw = rt_ & (wtr == w) & np.isfinite(p_oof)
            if mw.sum() > 50:
                per_well_r.append(float(np.nanmedian(ro[mw])))
                valid_w.append(w)
                Sw.append(S_tr[k])
        per_well_r = np.array(per_well_r)
        Sw = np.array(Sw)
        within = float(np.sqrt(np.nanmean(
            [np.nanvar(ro[rt_ & (wtr == w)]) for w in valid_w])))
        print(f"  A 训练井逐井残差 log10 中位：跨井 std {per_well_r.std():.4f}  "
              f"井内 std {within:.4f}  极差 {per_well_r.max() - per_well_r.min():+.3f}"
              f"  n_well {len(valid_w)}")

        # ---- B 生死判据：井级残差能否从井级统计量预测（留一井 LOO）----
        # 井级统计量含 DEPTH（量级 ~4000）等未缩放列，Ridge 必须逐折缩放，
        # 否则会由大尺度列主导、给出 MAE ≫ 目标 std 的假信号。
        best_name, best_mae, best_mse = None, 1e9, None
        if per_well_r.std() < 1e-3:
            print(f"     跨井 std {per_well_r.std():.4f} < 1e-3：井级偏差本身可忽略，"
                  f"LOO 无意义（无信号可预测）→ B/E 跳过")
            skip_B = True
        else:
            skip_B = False
        if skip_B:
            best_mae = 1e9
        else:
            for name, mk in (("Ridge", lambda: Ridge(alpha=1.0)),
                             ("HistGB", lambda: HistGradientBoostingRegressor(
                                 max_iter=200, max_leaf_nodes=31, min_samples_leaf=2,
                                 l2_regularization=5.0, random_state=42))):
                errs = []
                for i in range(len(valid_w)):
                    ii = np.ones(len(valid_w), bool)
                    ii[i] = False
                    mu = Sw[ii].mean(axis=0)
                    sd = np.where(Sw[ii].std(axis=0) < 1e-12, 1.0, Sw[ii].std(axis=0))
                    Xf = (Sw[ii] - mu) / sd
                    Xe = (Sw[[i]] - mu) / sd
                    mm = mk()
                    mm.fit(Xf, per_well_r[ii])
                    errs.append(float(mm.predict(Xe)[0] - per_well_r[i]))
                errs = np.abs(errs)
                mae = float(errs.mean())
                rmse = float(np.sqrt(np.mean(np.square(errs))))
                ok = mae < 0.5 * per_well_r.std()
                print(f"     {name:<7} 留一井 MAE {mae:.4f} / RMSE {rmse:.4f}"
                      f"  （残差 std {per_well_r.std():.4f}，比值 "
                      f"{mae / max(per_well_r.std(), 1e-9):.2f}）"
                      f"{'  ← 可用' if ok else ''}")
                if mae < best_mae:
                    best_mae, best_name, best_mse = mae, name, errs

        # ---- C 同特征二阶段（对照：残差能否用点级特征预测）----
        xo, rox = Xtr[rt_], ro[rt_]
        m2f = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_leaf_nodes=31,
                                            min_samples_leaf=80, l2_regularization=1.0,
                                            loss="squared_error", random_state=42)
        m2f.fit(xo, rox)
        r2 = m2f.predict(Xva[rv_])
        a2 = acc_vec(c, np.power(10.0, p_va - r2), np.power(10.0, tv)).mean()
        # base / a2 只在**真实段**上取均值，折算成总分必须再乘「真实段占有效点比例」，
        # 否则会被放大 1/占比 ≈ 2.3~4.4 倍。见 CLAUDE.md §4.1 的分解表（占比 POR 0.393 /
        # PERM 0.388 / SW 0.227，验证集口径）。
        frac = rv_.sum() / max(int(np.isfinite(yv).sum()), 1)
        print(f"  C 同特征二阶段残差校正：验证 log10-MAE {np.nanmean(np.abs(r2)):.4f}（一阶段 "
              f"{np.nanmean(np.abs(rr_va)):.4f}）→ 真实段 acc {base:.4f}→{a2:.4f}"
              f"   总分贡献 Δ {(a2 - base) * 100 * WEIGHTS[c] * frac:+.4f} 分"
              f"（真实段占比 {frac:.4f}；注意 base/a2 是真实段口径，×占比 才是总分）")

        # ---- D 官方口径带内/带外（带外梯度恒为 0 的直接后果）----
        tol = TOL_LOG[c]
        ib = np.abs(rr_va) <= tol
        out = ~ib
        print(f"  D 验证集带内 {ib.mean() * 100:5.1f}%（log10 容差 {tol:.5f}）"
              f"   带内 |残差| 中位 {np.median(np.abs(rr_va[ib])):.4f}"
              f"   带外 |残差| 中位 {np.median(np.abs(rr_va[out])):.4f}")

        # ---- E 若 B 可用：在验证井实测井级校正 ----
        if skip_B:
            best_mae = 1e9
            print("  E （B 已跳过：跨井偏差可忽略，井级校正无作用对象）")
        if best_mae < 0.5 * per_well_r.std():
            mk = Ridge if best_name == "Ridge" else (
                lambda: HistGradientBoostingRegressor(
                    max_iter=200, max_leaf_nodes=31, min_samples_leaf=2,
                    l2_regularization=5.0, random_state=42))
            mk2 = mk()
            mu, sd = Sw.mean(axis=0), np.where(Sw.std(axis=0) < 1e-12, 1.0, Sw.std(axis=0))
            mk2.fit((Sw - mu) / sd, per_well_r)
            cva = mk2.predict((S_va - mu) / sd)
            idx = np.flatnonzero(rv_)
            pt = np.array([cva[uwva.index(w)] for w in wva[idx]])
            for lam in (0.3, 0.5, 0.7, 1.0):
                a_new = acc_vec(c, np.power(10.0, p_va - lam * pt),
                                np.power(10.0, tv)).mean()
                print(f"     λ={lam:<4} 真实段 acc {base:.4f}→{a_new:.4f}"
                      f"   总分贡献 Δ {(a_new - base) * 100 * WEIGHTS[c] * frac:+.4f} 分")
            # 验证井残差预测 vs 真值
            pr, trr = [], []
            for i, w in enumerate(uwva):
                mw = rv_ & (wva == w)
                if mw.sum() > 50:
                    pr.append(float(cva[i]))
                    trr.append(float(np.nanmedian(rr_va[mw])))
            if len(pr) >= 3:
                print(f"     验证井残差预测 vs 真值 MAE {np.mean(np.abs(np.array(pr) - np.array(trr))):.4f}"
                      f"  相关 {np.corrcoef(pr, trr)[0, 1]:+.3f}  真实残差 std {np.std(trr):.4f}")
        else:
            print(f"  E 井级残差**不可从井级统计量预测**（{best_name} 留一井 MAE {best_mae:.4f}"
                  f" ≥ 0.5×std {0.5 * per_well_r.std():.4f}）→ 方向关闭")

        del m_full, m2f
        gc.collect()

    print(f"\n用时 {time.time() - t0:.0f}s   峰值 {rss()}")


if __name__ == "__main__":
    main()
