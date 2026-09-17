"""基线诊断：分数分解 + oracle 上界 + 决策规则对比（只读分析，不改动模型文件）."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.feature import (
    default_mask, inverse_transform_target, load_wells, make_features,
    make_targets, transform_target,
)
from src.model import TwoStageModel
from src.utils import DEFAULT_VALUES, EPS, TARGETS, TOLERANCE, WEIGHTS


def pointwise(c: str, pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    p = np.asarray(pred, float)
    t = np.asarray(true, float)
    if c == "PERM":
        p = np.log10(np.maximum(p, EPS))
        t = np.log10(np.maximum(t, EPS))
    denom = TOLERANCE[c] * (np.abs(t) + EPS)
    return np.maximum(0.0, 1.0 - np.abs(p - t) / denom)


def acc(c: str, pred: np.ndarray, true: np.ndarray) -> float:
    m = ~np.isnan(true)
    return float(pointwise(c, pred[m], true[m]).mean())


def main() -> None:
    wells = load_wells("./train", has_labels=True)
    ids = [w["logId"].iloc[0] for w in wells]
    tr = [w for w in wells if w["logId"].iloc[0] in set(ids[:72])]
    va = [w for w in wells if w["logId"].iloc[0] in set(ids[72:])]
    tr_df = pd.concat(tr, ignore_index=True)
    va_df = pd.concat(va, ignore_index=True)
    print(f"train wells={len(tr)} rows={len(tr_df):,}  val wells={len(va)} rows={len(va_df):,}")

    Xtr, Ytr = make_features(tr_df), make_targets(tr_df)
    Xva, Yva = make_features(va_df), make_targets(va_df)
    dtr, dva = default_mask(Ytr), default_mask(Yva)

    model = TwoStageModel().fit(Xtr, Ytr, dtr)
    print("fit done")

    # 验证集上调阈值（诊断用）
    p_def = {c: model.classifiers[c].predict_proba(Xva)[:, 1] for c in TARGETS}
    reg_phys = {
        c: inverse_transform_target(c, pd.Series(model.regressors[c].predict(Xva), index=Xva.index)).to_numpy()
        for c in TARGETS
    }
    th = {}
    for c in TARGETS:
        y = Yva[c].to_numpy()
        m = ~np.isnan(y)
        best = (0.5, -1.0)
        for t in np.arange(0.05, 0.96, 0.05):
            pred = np.where(p_def[c] >= t, DEFAULT_VALUES[c], reg_phys[c])
            a = pointwise(c, pred[m], y[m]).mean()
            if a > best[1]:
                best = (float(t), float(a))
        th[c] = best[0]
        print(f"{c}: tuned th={best[0]:.2f} acc={best[1]:.4f}")

    print("\n=== 分数分解（验证集 8 井）===")
    total = 0.0
    for c in TARGETS:
        y = Yva[c].to_numpy()
        valid = ~np.isnan(y)
        isdef = dva[c].to_numpy() & valid
        real = valid & ~isdef
        pred = np.where(p_def[c] >= th[c], DEFAULT_VALUES[c], reg_phys[c])
        a_all = pointwise(c, pred[valid], y[valid]).mean()
        a_def = pointwise(c, pred[isdef], y[isdef]).mean() if isdef.sum() else 0.0
        a_real = pointwise(c, pred[real], y[real]).mean() if real.sum() else 0.0
        a_real_ifdef = pointwise(c, np.full(real.sum(), DEFAULT_VALUES[c]), y[real]).mean() if real.sum() else 0.0
        # oracle 分类 + 当前回归
        pred_oc = np.where(isdef, DEFAULT_VALUES[c], reg_phys[c])
        a_oc = pointwise(c, pred_oc[valid], y[valid]).mean()
        fd, fr = isdef.sum() / valid.sum(), real.sum() / valid.sum()
        total += WEIGHTS[c] * a_all
        print(
            f"{c}: acc={a_all:.4f} | 默认段占比={fd:.3f} acc_def={a_def:.4f} | "
            f"真实段占比={fr:.3f} acc_real={a_real:.4f} (若输出默认={a_real_ifdef:.4f}) | "
            f"oracle分类+当前回归={a_oc:.4f}"
        )
    print(f"总分(验证)={total * 100:.3f}")

    print("\n=== 真实段 oracle 上界（沿深度持久性 / 窗口中位数）===")
    well_ids = va_df["logId"].to_numpy()
    for c in TARGETS:
        pers, wm, regv = [], [], []
        for w in va:
            lid = w["logId"].iloc[0]
            wmask = well_ids == lid
            Yw = make_targets(w)
            y = Yw[c].to_numpy()
            isd = default_mask(Yw)[c].to_numpy()
            real = ~np.isnan(y) & ~isd
            yp = np.roll(y, 1)
            okp = np.roll(real, 1)
            okp[0] = False
            sel = real & okp
            pers.append(pointwise(c, yp[sel], y[sel]))
            regv.append(pointwise(c, reg_phys[c][wmask][sel], y[sel]))
            idx = np.where(real)[0]
            wmarr = np.full_like(y, np.nan)
            for i in idx:
                lo = max(0, i - 5)
                seg = y[lo : i + 1]
                seg = seg[~np.isnan(seg)]
                wmarr[i] = float(np.median(seg)) if seg.size else float(y[i])
            wm.append(pointwise(c, wmarr[sel], y[sel]))
        pers = np.concatenate(pers); wm = np.concatenate(wm); regv = np.concatenate(regv)
        print(f"{c}: 真实段 n={pers.size:,}  持久性y[t-1]={pers.mean():.4f}  因果窗口中位数={wm.mean():.4f}  当前回归={regv.mean():.4f}")

    print("\n=== 决策规则对比：硬阈值 vs 期望分数最大化 ===")
    rng = np.random.default_rng(0)
    sigma = {}
    for c in TARGETS:
        ytr = Ytr[c]
        real_tr = ytr.notna() & ~dtr[c]
        rt = transform_target(c, ytr[real_tr]).to_numpy()
        pt = model.regressors[c].predict(Xtr[real_tr.to_numpy()])
        sigma[c] = float(np.std(rt - pt))
        print(f"{c}: 真实段残差 sigma(transform)={sigma[c]:.4f}")
    total_hard, total_exp = 0.0, 0.0
    for c in TARGETS:
        y = Yva[c].to_numpy()
        m = ~np.isnan(y)
        n = int(m.sum())
        p = p_def[c]
        reg_t = transform_target(c, pd.Series(reg_phys[c])).to_numpy()[m]
        draws = reg_t[:, None] + sigma[c] * rng.standard_normal((n, 16))
        if c == "PERM":
            draws = np.power(10.0, draws)
        cand = [np.full(n, DEFAULT_VALUES[c]), reg_phys[c][m]]
        for a in np.linspace(0.15, 0.85, 5):
            if c == "PERM":
                blend = 10 ** ((1 - a) * np.log10(max(DEFAULT_VALUES[c], EPS)) + a * np.log10(np.maximum(reg_phys[c][m], EPS)))
            else:
                blend = (1 - a) * DEFAULT_VALUES[c] + a * reg_phys[c][m]
            cand.append(blend)
        cand = np.stack(cand)  # (K, N)
        def_part = pointwise(c, cand, DEFAULT_VALUES[c]).T  # (N, K)
        mc = np.stack([pointwise(c, cand, draws[:, j]) for j in range(draws.shape[1])]).mean(0)  # (K, N)
        exp_score = p[m][:, None] * def_part + (1 - p[m])[:, None] * mc.T  # (N, K)
        best_c = cand[np.argmax(exp_score, 1), np.arange(n)]
        hard = np.where(p >= th[c], DEFAULT_VALUES[c], reg_phys[c])[m]
        a_hard = pointwise(c, hard, y[m]).mean()
        a_exp = pointwise(c, best_c, y[m]).mean()
        total_hard += WEIGHTS[c] * a_hard
        total_exp += WEIGHTS[c] * a_exp
        print(f"{c}: 硬阈值={a_hard:.4f}  期望分数决策={a_exp:.4f}  delta={a_exp - a_hard:+.4f}")
    print(f"总分: 硬阈值={total_hard * 100:.3f}  期望决策={total_exp * 100:.3f}")


if __name__ == "__main__":
    main()
