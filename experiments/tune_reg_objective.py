"""回归器训练目标形状实验：树回归器该优化什么损失？

核心观察（此前从未试过，因为代码里没有这个开关）
------------------------------------------------
官方三目标折算到 log10 空间后都是「最小化 |log10(ŷ/y)|」，带分别为
  POR 0.03342 / PERM 1.0 / SW 0.02119
即都是**对数绝对误差**，只有一个常数带宽。但现状是：

  src/model.py::_reg() 硬编码 loss="squared_error"（且 p.update 之后又传 loss=... 会
  TypeError，所以**仓库里没有任何代码路径能用别的目标函数训练树**）。
  PERM 因 target_transform="perm_log" 在 log10 空间训练 —— squared_error 在这里是
  对数均方误差，不是官方口径。
  POR / SW 在**线性空间**用 squared_error 训练 —— 既不是对数，也不是绝对误差。
  再叠加 weight_tol 的样本权重 w=|y|（POR/SW），把大 y 的点权重再抬高，与官方
  「相对误差、逐点等权」的方向正好相反。

已排除 / 需注意的边界
---------------------
  * PERM 换 log10-MAE 已试过，只值 +0.003 real（CLAUDE.md §4.6）→ 本脚本保留 PERM 现状，
    只动 POR / SW 的目标形状。
  * 「三目标 all_log」在 §7 被否决，但那是**旧 PERM 口径**下的结论；官方口径把 PERM
    分数整体抬高 4.5 分、把整张分数地图都换了（§11），所以这条需要按官方口径重验。
  * 分段器不动、融合权重不动、TCN 不重训（context=False，纯 numpy 前向，零 GPU）。

口径纪律
--------
  1) V0 基线必须用 eval_v4.build_tree 原路径拟合，并与 screen_tree_cache 逐点比对
     （中位差 < 1e-6），否则 §4.10 的陷阱会把增益调在错误的回归器上。
  2) 每个变体与基线共用**同一套 TCN 输出、同一套分段器 p_def**，唯一变量是树回归器。
  3) 报告 acc / acc_def / acc_real 分解 + 逐井 Δ 的留一井包络。
  4) v9 阈值 {0.080, 0.250, 0.050} 是主口径；另给逐目标阈值重扫的最优值，用于判断
     增益是不是「更差的回归器 × 更大的阈值」的联合过拟合（§4.11 的指纹）。

用法: python experiments/tune_reg_objective.py
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
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from src.feature import (
    load_wells, make_targets, default_mask, make_features,
    transform_target, inverse_transform_target,
)
from src.utils import TARGETS, WEIGHTS, EPS, DEFAULT_VALUES
from src.model import DEFAULT_CLF_PARAMS, DEFAULT_REG_PARAMS, POR_MONOTONIC
from experiments.eval_v4 import build_tree, tcn_outputs

RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
TAGS = ["T15_seed42_150ep", "T16_seed43_150ep", "T16_seed44_150ep"]
CFG = {"blend_w": {"POR": 0.50, "PERM": 0.50, "SW": 0.75},
       "thresholds": {"POR": 0.080, "PERM": 0.250, "SW": 0.050},
       "clf_tcn_w": {"POR": 0.60, "PERM": 0.70, "SW": 0.30}}
TH_V7 = {"POR": 0.130, "PERM": 0.270, "SW": 0.050}
GRID = np.arange(0.02, 0.71, 0.005)
# log10 空间容差：|log10(ŷ/y)| ≤ log10(1+tol)
TOL_LOG = {"POR": float(np.log10(1.08)), "PERM": 1.0, "SW": float(np.log10(1.05))}


def rss() -> str:
    import resource
    return f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576:.0f}MB"


def acc_vec(c, p, t):
    p = np.asarray(p, float)
    t = np.asarray(t, float)
    if c == "PERM":
        return np.maximum(0.0, 1.0 - np.abs(np.log10(np.maximum(p / np.maximum(t, EPS), EPS))))
    tol = 0.08 if c == "POR" else 0.05
    return np.maximum(0.0, 1.0 - np.abs(p - t) / (tol * (np.abs(t) + EPS)))


def fit_variant(Xtr: pd.DataFrame, Ytr: pd.DataFrame, mode: str, loss: str,
                weight: str) -> dict:
    """逐目标拟合（分类器原样，回归器按 mode/loss/weight 定制）。

    mode:   "perm_log" = 仅 PERM 取 log10（部署现状）；"all_log" = 三目标全 log10
    loss:   sklearn HistGB 的 loss 名
    weight: "abs_y"   → w = |y|          （部署现状 POR/SW）
            "abslogy" → w = |log10 y|    （与 log10 目标一致的加权）
            "none"    → 不做 weight_tol 二次拟合
    """
    Dtr = default_mask(Ytr)
    out = {}
    for c in TARGETS:
        valid = Ytr[c].notna().to_numpy()
        y = Ytr.loc[valid, c]
        def_ = Dtr.loc[valid, c].astype(int)
        Xv = Xtr.loc[valid]
        real = (def_ == 0).to_numpy()

        clf = HistGradientBoostingClassifier(loss="log_loss", random_state=42, **DEFAULT_CLF_PARAMS)
        clf.fit(Xv, def_)

        mono = POR_MONOTONIC if c == "POR" else None
        cols = list(Xv.columns)
        cst = [mono.get(col, 0) for col in cols] if mono else None
        params = dict(DEFAULT_REG_PARAMS)
        params["loss"] = loss
        if cst is not None and any(cst):
            params["monotonic_cst"] = cst
        reg = HistGradientBoostingRegressor(random_state=42, **params)
        yt = transform_target(c, y.loc[real], mode)
        try:
            reg.fit(Xv.loc[real], yt)
        except Exception as e:                      # gamma / poisson 需要严格正值
            print(f"        {c} 拟合失败 ({loss}): {type(e).__name__}: {str(e)[:60]}")
            return None

        if weight != "none":
            yv = y.to_numpy()[real]
            w = (np.abs(np.log10(np.maximum(yv, EPS))) if weight == "abslogy"
                 else np.abs(yv))
            w = np.clip(w, np.percentile(w, 5), np.percentile(w, 95))
            w = w / w.mean()
            reg.fit(Xv.loc[real], yt, sample_weight=w)
        out[c] = dict(clf=clf, reg=reg)
    return out


def main():
    t0 = time.time()
    wells = load_wells("./train", has_labels=True)
    ids = [w["logId"].iloc[0] for w in wells]
    by = dict(zip(ids, wells))
    tr72 = [by[i] for i in ids[:72]]
    va8 = [by[i] for i in ids[72:80]]
    tr = pd.concat(tr72, ignore_index=True)
    va = pd.concat(va8, ignore_index=True)
    del wells, tr72, va8
    gc.collect()

    Xtr = make_features(tr, context=True, priors=True)
    Xva = make_features(va, context=True, priors=True)
    Ytr, Yva = make_targets(tr), make_targets(va)
    Dtr, Dva = default_mask(Ytr), default_mask(Yva)
    print(f"训练 {len(tr):,} 点 / 72 井   验证 {len(va):,} 点 / 8 井   特征 {Xtr.shape[1]}")
    print(f"log10 容差 " + "  ".join(f"{c} {TOL_LOG[c]:.5f}" for c in TARGETS))
    print(f"γ/POR 严格正值检查：POR 最小 {Ytr['POR'].min():.4f}  SW 最小 {Ytr['SW'].min():.4f}"
          f"   {'✓' if Ytr['POR'].min() > 0 and Ytr['SW'].min() > 0 else '✗ gamma 不可用'}")

    arts = [joblib.load(os.path.join(RUNS, f"tcn_{t}.joblib")) for t in TAGS]
    va_wells = [by[i] for i in ids[72:80]]
    ens, pcn = {}, {}
    for art in arts:
        r, p = tcn_outputs(va_wells, art)
        for c in TARGETS:
            ens.setdefault(c, []).append(r[c])
            pcn.setdefault(c, []).append(p[c])
    ens = {c: np.mean(v, axis=0) for c, v in ens.items()}
    pcn = {c: np.mean(v, axis=0) for c, v in pcn.items()}

    def score(reg_tree, pt_tree, ths, tag=""):
        """reg_tree: {c: log10 预测}；pt_tree: {c: p_def}。"""
        tot, parts, th_used = 0.0, {}, {}
        for c in TARGETS:
            y = Yva[c].to_numpy().astype(float)
            valid = np.isfinite(y)
            reg = (1 - CFG["blend_w"][c]) * reg_tree[c] + CFG["blend_w"][c] * ens[c]
            reg = np.where(np.isfinite(reg), reg, -5.0)
            cw = CFG["clf_tcn_w"][c]
            pm = (1 - cw) * pt_tree[c] + cw * pcn[c]
            pred = np.where(pm >= ths[c], DEFAULT_VALUES[c], np.power(10.0, reg))
            a = np.where(np.isfinite(y), acc_vec(c, pred, y), np.nan)
            tot += WEIGHTS[c] * np.nanmean(a)
            d = Dva[c].to_numpy() & valid
            parts[c] = dict(acc=float(np.nanmean(a)),
                            real=float(np.nanmean(a[valid & ~d])) if (valid & ~d).sum() else float("nan"),
                            defv=float(np.nanmean(a[valid & d])) if (valid & d).sum() else float("nan"),
                            pm=float(pm[valid].mean()),
                            a=a, valid=valid, d=d)
            th_used[c] = ths[c]
        return tot * 100, parts, th_used

    # ---------- V0 基线：部署原路径 + 与缓存逐点比对 ----------
    print("\n" + "=" * 96)
    print("V0 基线（eval_v4.build_tree 原路径 = 部署配置）")
    print("=" * 96)
    model, _ = build_tree(tr, weight_tol=True)
    reg0 = {c: np.log10(np.maximum(inverse_transform_target(
        c, pd.Series(model.regressors[c].predict(Xva), index=Xva.index)).to_numpy(), EPS))
        for c in TARGETS}
    pt0 = {c: model.classifiers[c].predict_proba(Xva)[:, 1] for c in TARGETS}

    cache = joblib.load(os.path.join(RUNS, "screen_tree_cache.joblib"))
    for c in TARGETS:
        a = float(np.median(np.abs(np.sort(reg0[c]) - np.sort(cache["reg"][c]))))
        b = float(np.median(np.abs(np.sort(pt0[c]) - np.sort(cache["p_def"][c]))))
        print(f"  [参考] {c} 与 screen_tree_cache 树 reg 中位差 {a:.2e} / p_def {b:.2e}"
              "（§4.10 已记录此口径存在 ~1e-3~1e-2 的未追根因残差，仅作诊断、不作门控）")
    t_v9, d_v9, _ = score(reg0, pt0, CFG["thresholds"])
    t_v7, d_v7, _ = score(reg0, pt0, TH_V7)
    print(f"  [自检] Total：v9 阈值 {t_v9:.3f}（应复现记录的 v9 screen 73.931）"
          f"   v7 阈值 {t_v7:.3f}（225 特征口径下才是 73.749，此处 237 特征不可比）")
    print("  [自检] v9 口径分解 " + "  ".join(
        f"{c} acc {d_v9[c]['acc']:.4f}/real {d_v9[c]['real']:.4f}/def {d_v9[c]['defv']:.4f}"
        for c in TARGETS))
    # 真正的门控：基线必须复现项目自己记录的 v9 harness screen 分数。这比逐点比对缓存
    # 强得多 —— 它校验整条流水线（树 + 3 TCN + 融合 + 阈值 + 口径），而缓存逐点差是
    # §4.10 已知的历史残差，与流水线正确性无关。
    base = t_v9
    if abs(base - 73.931) > 0.02:
        print(f"  ✗ 基线 {base:.3f} 未能复现 v9 screen 73.931 → 退出（口径或特征集已变）")
        return
    print(f"  ✓ 基线复现成功，以此为唯一比较基准   {rss()}")

    # ---------- 变体：只动 POR / SW 的 (变换, 损失, 加权) ----------
    VARIANTS = [
        ("V1 all_log + squared_error  (all_log 在官方口径下重验)", "all_log", "squared_error", "abs_y"),
        ("V2 all_log + absolute_error (官方口径形状，全部目标)",    "all_log", "absolute_error", "abs_y"),
        ("V3 all_log + abs_error + w=|log10 y| (加权与目标一致)",  "all_log", "absolute_error", "abslogy"),
        ("V4 all_log + absolute_error 无 weight_tol",              "all_log", "absolute_error", "none"),
        ("V5 linear + absolute_error (只对数化的对照组)",          "perm_log", "absolute_error", "abs_y"),
        ("V6 linear + gamma (正值目标的相对误差损失)",            "perm_log", "gamma", "abs_y"),
        ("V7 linear + poisson (正值目标的泊松偏差)",              "perm_log", "poisson", "abs_y"),
    ]
    # PERM 在 perm_log 下本来就是 log10；V2/V3/V4 的 all_log 让 POR/SW 也进 log10。
    print("\n" + "=" * 96)
    print("变体（唯一变量 = 树回归器的 (目标变换, 损失, 样本加权)；分段器与 TCN 共用")
    print("     报告：v9 阈值下 Total 与分解；再逐目标重扫阈值判断是否联合过拟合")
    print("=" * 96)

    results = []
    for name, mode, loss, weight in VARIANTS:
        t = time.time()
        got = fit_variant(Xtr, Ytr, mode, loss, weight)
        if got is None:
            print(f"\n{name}\n        放弃（某目标拟合失败，见上）")
            continue
        # ⚠️ 逆变换必须用**本变体的 mode**，不能用默认的 perm_log：
        # all_log 下 POR/SW 的树输出已经是 log10，若再套一次 perm_log 逆变换会原样返回，
        # 下面外层那个 np.log10 就变成二次取对数 → POR/SW 预测全废（实测 real acc 塌到 0.002）。
        reg_t = {c: np.log10(np.maximum(inverse_transform_target(
            c, pd.Series(got[c]["reg"].predict(Xva), index=Xva.index), mode).to_numpy(), EPS))
            for c in TARGETS}
        pt_t = {c: got[c]["clf"].predict_proba(Xva)[:, 1] for c in TARGETS}

        # 健全性检查：log10 空间预测必须落在基线的动态范围内。二次取对数会把
        # log10(POR)≈-1 变成 log10(-1)=NaN，再被 -5.0 哨兵吃掉 —— 那次事故静默损失了 40 分钟。
        sane = True
        for c in TARGETS:
            b0, nt = reg0[c], reg_t[c]
            lo, hi = float(np.percentile(b0, 1)), float(np.percentile(b0, 99))
            frac_out = float(np.mean((nt < lo - 0.5) | (nt > hi + 0.5)))
            if frac_out > 0.05:
                print(f"        ✗ {c} 预测 {frac_out * 100:.1f}% 超出基线动态范围 "
                      "[1%,99%]={:.2f},{:.2f} → 逆变换 mode 不匹配，放弃该变体".format(lo, hi))
                sane = False
        if not sane:
            del got
            gc.collect()
            continue

        tv, dv, _ = score(reg_t, pt_t, CFG["thresholds"])
        d = tv - base
        print(f"\n{name}\n  v9 阈值 Total {base:.3f} → {tv:.3f}   Δ {d:+.3f} 分"
              f"{'   ← ≥+0.10' if d >= 0.10 else ''}   ({time.time() - t:.0f}s)")
        for c in TARGETS:
            b = d_v9[c]
            print(f"     {c}  acc {b['acc']:.4f}→{dv[c]['acc']:.4f}"
                  f"   real {b['real']:.4f}→{dv[c]['real']:.4f}"
                  f"   def {b['defv']:.4f}→{dv[c]['defv']:.4f}"
                  f"   Δ {(dv[c]['acc'] - b['acc']) * 100 * WEIGHTS[c]:+.4f} 分")

        # 逐目标阈值重扫（在 v9 阈值不动其他目标的前提下）
        th = dict(CFG["thresholds"])
        best = tv
        for c in TARGETS:
            for g in GRID:
                th2 = dict(CFG["thresholds"])
                th2[c] = g
                b2, _, _ = score(reg_t, pt_t, th2)
                if b2 > best:
                    best, th = b2, th2
        print(f"     重扫最优 Total {base:.3f} → {best:.3f}   Δ {best - base:+.3f} 分   "
              f"阈值 {{{' '.join(f'{c}={th[c]:.3f}' for c in TARGETS)}}}"
              f"   （v9 阈值下已是 {tv:.3f}，重扫只多 {best - tv:+.3f}）")
        results.append(dict(name=name, mode=mode, loss=loss, weight=weight,
                            v9=tv, opt=best, d_v9=d, d_opt=best - base, th=th,
                            reg_t=reg_t, pt_t=pt_t))
        del got
        gc.collect()

    # ---------- 排名与真实段错误诊断 ----------
    print("\n" + "=" * 96)
    print("排名（v9 阈值 / 重扫最优）")
    print("=" * 96)
    print(f"  {'变体':<52}{'v9 阈值':>9}{'Δv9':>8}{'重扫最优':>9}{'Δopt':>8}{'重扫增益':>9}")
    print(f"  {'V0 基线 (perm_log, squared_error, w=|y|)':<52}{base:>9.3f}{'':>8}{base:>9.3f}"
          f"{'':>8}{'':>9}")
    for r in sorted(results, key=lambda x: -x["d_v9"]):
        print(f"  {r['name']:<52}{r['v9']:>9.3f}{r['d_v9']:>+8.3f}{r['opt']:>9.3f}"
              f"{r['d_opt']:>+8.3f}{r['d_opt'] - r['d_v9']:>+9.3f}")

    print("\n" + "=" * 96)
    print("真实段 log10 误差诊断（决定哪个变体离容差带更近；带越窄越需要把 MAE 压小）")
    print("=" * 96)
    print(f"  {'变体':<40}" + "  ".join(
        f"{c:>26}" for c in TARGETS) + "   带内占比")
    names = ["V0 基线"] + [r["name"].split(" (")[0] for r in sorted(results, key=lambda x: -x["v9"])]
    regs = [reg0] + [r["reg_t"] for r in sorted(results, key=lambda x: -x["v9"])]
    for name, rt in zip(names, regs):
        bits = []
        inband = []
        for c in TARGETS:
            y = Yva[c].to_numpy().astype(float)
            reg = (1 - CFG["blend_w"][c]) * rt[c] + CFG["blend_w"][c] * ens[c]
            reg = np.where(np.isfinite(reg), reg, -5.0)
            m = np.isfinite(y) & ~Dva[c].to_numpy()
            # ⚠️ reg 是全长 (72779)，必须用 m 索引后才与 y[m] 同长
            e = np.abs(reg[m] - np.log10(np.clip(y[m], 1e-12, None)))
            bits.append(f"MAE {np.mean(e):.4f}/中位 {np.median(e):.4f}")
            inband.append(f"{(e <= TOL_LOG[c]).mean() * 100:5.1f}%")
        print(f"  {name:<40}" + "  ".join(f"{b:>26}" for b in bits)
              + "   " + "  ".join(f"{c}:{v}" for c, v in zip(TARGETS, inband)))

    print(f"\n总用时 {time.time() - t0:.0f}s   峰值 {rss()}")


if __name__ == "__main__":
    main()
