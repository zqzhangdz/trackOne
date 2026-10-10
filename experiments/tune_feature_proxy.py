"""追加「此前完全未编码的维度」：物理代理 + 井内秩 + 归一化井深。

为什么值得试
------------
CLAUDE.md §4.14 确立的项目唯一有效原则：**追加一个此前完全未编码的维度**有效
（逐井 z 分数 225→237，harness +0.171），而把已有的 13 条曲线切得更碎无效
（±60 上下文 225→321 是负资产）。所以候选必须是「结构上不存在」的信息。

对现有 237 特征的完整清点后，发现一个明确的结构性缺口：

  add_physics_priors 给的 4 个先验 —— POR_D、POR_S、POR_avg、POR_ds ——
  **全部是 POR 的代理**。PERM 与 SW 没有任何物理代理特征：
    * SW 有标准 Archie 关系（含孔隙度与电阻率），但仓库里只用了
      RT、RXO、logRT、logRXO、RT/RXO，从未组过 Archie 形式。
    * PERM 的经典 Kozeny-Carman 关系 k ∝ φ³/(1−φ)² 从未编码。
    * §7 否决的是「物理公式**约束**」（对输出做硬裁剪），不是「物理导出的**输入**」
      —— 后者恰恰是现有 4 个 POR 先验在做的事。
  另有两条未编码的井级维度：
    * 井深归一化位置（rel_depth / 本井井深跨度）：各井井深不同，储层层位可能在
      相对深度上可对齐。rel_depth 是绝对米数，从未归一化。
    * 井内秩 / 累积分布位置：wellz 是 (x−mean)/std（对异常值敏感、无界），
      而「本点在**本井**这条曲线分布中的分位」是另一个编码（有界 [0,1]、对异常值稳健），
      与 wellz 不是同一个变换。对默认段尤其相关：默认段取值恒定，秩会大量并列。

口径纪律：唯一变量是特征集；特征变了**分类器与回归器必须一起重拟合**；TCN 走
context=False 不受影响、零 GPU；v9 阈值为主口径，另逐目标重扫判断是否联合过拟合。

用法: python experiments/tune_feature_proxy.py
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
from src.model import DEFAULT_CLF_PARAMS, DEFAULT_REG_PARAMS, POR_MONOTONIC
from src.utils import TARGETS, WEIGHTS, EPS, DEFAULT_VALUES
from experiments.eval_v4 import build_tree, tcn_outputs

RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
TAGS = ["T15_seed42_150ep", "T16_seed43_150ep", "T16_seed44_150ep"]
CFG = {"blend_w": {"POR": 0.50, "PERM": 0.50, "SW": 0.75},
       "thresholds": {"POR": 0.080, "PERM": 0.250, "SW": 0.050},
       "clf_tcn_w": {"POR": 0.60, "PERM": 0.70, "SW": 0.30}}
GRID = np.arange(0.02, 0.71, 0.005)
RANK_CURVES = ["DEN", "AC", "CNL", "GR", "SP", "RT"]


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


def add_proxy(X: pd.DataFrame, df: pd.DataFrame, with_proxy=True,
              with_depth=True, with_rank=True) -> pd.DataFrame:
    """追加物理代理 + 井深归一化 + 井内秩（全部只依赖输入曲线，推理可用）。"""
    X = X.copy()
    gid = df["logId"].to_numpy()
    dep = df["DEPTH"].to_numpy(dtype=float)

    if with_proxy:
        DEN, CNL = X["DEN"].to_numpy(), X["CNL"].to_numpy()
        RT = X["RT"].to_numpy()
        # PERM 代理：Kozeny-Carman k ∝ φ³/(1−φ)²
        phi_d = np.clip((2.65 - DEN) / 1.65, 1e-4, 0.99)
        X["PERM_kc"] = np.log10(np.clip(phi_d ** 3 / (1.0 - phi_d) ** 2, 1e-12, None))
        # SW 代理：Archie 形式，Sw ∝ (φ / RT)^(1/n)
        phi_n = np.clip(CNL / 100.0, 1e-4, 0.99)
        lrt = np.log10(np.clip(RT, 1e-6, None))
        X["SW_archie_n2"] = np.log10(np.clip(phi_n / 2.0 / 10.0 ** lrt, 1e-9, None))
        X["SW_archie_n1"] = np.log10(np.clip(phi_n / 10.0 ** lrt, 1e-9, None))
        X["SW_phi_over_rt"] = np.log10(np.clip(phi_n, 1e-9, None)) - lrt

    if with_depth:
        dn = np.zeros(len(df))
        for g in np.unique(gid):
            m = gid == g
            d = dep[m]
            sp = d.max() - d.min()
            dn[m] = (d - d.min()) / sp if (np.isfinite(sp) and sp > 0) else np.nan
        X["depth_norm"] = dn

    if with_rank:
        for c in RANK_CURVES:
            v = X[c].to_numpy(dtype=float)
            col = np.full(len(df), np.nan)
            for g in np.unique(gid):
                m = gid == g
                col[m] = pd.Series(v[m]).rank(method="average", pct=True).to_numpy()
            X[f"rank_{c}"] = col
    return X


def fit_two_stage(Xtr: pd.DataFrame, Ytr: pd.DataFrame) -> dict:
    """与部署一致的逐目标两阶段拟合（分类器 + 真实段回归器 + weight_tol 重拟合）。"""
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
        if cst is not None and any(cst):
            params["monotonic_cst"] = cst
        reg = HistGradientBoostingRegressor(random_state=42, **params)
        yt = transform_target(c, y.loc[real], "perm_log")
        reg.fit(Xv.loc[real], yt)

        yv = y.to_numpy()[real]
        w = (np.abs(np.log10(np.maximum(yv, EPS))) if c == "PERM" else np.abs(yv))
        w = np.clip(w, np.percentile(w, 5), np.percentile(w, 95))
        w = w / w.mean()
        reg.fit(Xv.loc[real], yt, sample_weight=w)
        out[c] = dict(clf=clf, reg=reg)
    return out


def score(clf_map, reg_map, Xva, ens, pcn, Yva, Dva, ths=None):
    ths = ths or CFG["thresholds"]
    tot, parts, reg_tree, pt_tree = 0.0, {}, {}, {}
    for c in TARGETS:
        reg_tree[c] = np.log10(np.maximum(inverse_transform_target(
            c, pd.Series(reg_map[c].predict(Xva), index=Xva.index)).to_numpy(), EPS))
        pt_tree[c] = clf_map[c].predict_proba(Xva)[:, 1]
        y = Yva[c].to_numpy().astype(float)
        valid = np.isfinite(y)
        reg = (1 - CFG["blend_w"][c]) * reg_tree[c] + CFG["blend_w"][c] * ens[c]
        reg = np.where(np.isfinite(reg), reg, -5.0)
        cw = CFG["clf_tcn_w"][c]
        pm = (1 - cw) * pt_tree[c] + cw * pcn[c]
        pred = np.where(pm >= ths[c], DEFAULT_VALUES[c], np.power(10.0, reg))
        a = np.where(valid, acc_vec(c, pred, y), np.nan)
        tot += WEIGHTS[c] * np.nanmean(a)
        d = Dva[c].to_numpy() & valid
        parts[c] = dict(acc=float(np.nanmean(a)),
                        real=float(np.nanmean(a[valid & ~d])) if (valid & ~d).sum() else float("nan"),
                        defv=float(np.nanmean(a[valid & d])) if (valid & d).sum() else float("nan"))
    return tot * 100, parts, reg_tree, pt_tree


def main():
    t0 = time.time()
    wells = load_wells("./train", has_labels=True)
    ids = [w["logId"].iloc[0] for w in wells]
    by = dict(zip(ids, wells))
    tr = pd.concat([by[i] for i in ids[:72]], ignore_index=True)
    va = pd.concat([by[i] for i in ids[72:80]], ignore_index=True)
    va_wells = [by[i] for i in ids[72:80]]
    Ytr, Yva = make_targets(tr), make_targets(va)
    Dva = default_mask(Yva)
    del wells
    gc.collect()

    arts = [joblib.load(os.path.join(RUNS, f"tcn_{t}.joblib")) for t in TAGS]
    ens, pcn = {}, {}
    for art in arts:
        r, p = tcn_outputs(va_wells, art)
        for c in TARGETS:
            ens.setdefault(c, []).append(r[c])
            pcn.setdefault(c, []).append(p[c])
    ens = {c: np.mean(v, axis=0) for c, v in ens.items()}
    pcn = {c: np.mean(v, axis=0) for c, v in pcn.items()}
    print(f"训练 {len(tr):,} 点 / 72 井   验证 {len(va):,} 点 / 8 井")

    # ---------- 基线：部署原路径 + 缓存自检 ----------
    Xva0 = make_features(va, context=True, priors=True)
    m0, _ = build_tree(tr, weight_tol=True)
    cache = joblib.load(os.path.join(RUNS, "screen_tree_cache.joblib"))
    r0 = {c: m0.regressors[c] for c in TARGETS}
    c0 = {c: m0.classifiers[c] for c in TARGETS}
    ok = True
    for c in TARGETS:
        rt = np.log10(np.maximum(inverse_transform_target(
            c, pd.Series(r0[c].predict(Xva0), index=Xva0.index)).to_numpy(), EPS))
        pt = c0[c].predict_proba(Xva0)[:, 1]
        a = float(np.median(np.abs(np.sort(rt) - np.sort(cache["reg"][c]))))
        b = float(np.median(np.abs(np.sort(pt) - np.sort(cache["p_def"][c]))))
        ok &= max(a, b) < 1e-6
        print(f"  [自检] {c} reg 中位差 {a:.2e} / p_def {b:.2e}"
              f"{'   ✓' if max(a, b) < 1e-6 else '   ✗'}")
    t_base, d_base, _, _ = score(c0, r0, Xva0, ens, pcn, Yva, Dva)
    # 门控是「复现项目自己记录的 v9 screen 73.931」，不是逐点比对缓存：后者有 §4.10
    # 记录的 ~1e-3~1e-2 历史残差，与流水线正确性无关；而 Total 复现校验整条链路。
    print(f"  [自检] 基线 Total {t_base:.3f}（应复现记录的 v9 screen 73.931）"
          f"   {'✓ 复现成功' if abs(t_base - 73.931) <= 0.02 else '✗ 未能复现'}")
    if abs(t_base - 73.931) > 0.02:
        print("  ✗ 基线未复现，退出（口径或特征集已变）")
        return
    print(f"  基线特征数 {Xva0.shape[1]}   {rss()}")

    # ---------- 变体 ----------
    GROUPS = [
        ("G1 物理代理 3 列 (PERM_kc, SW_archie_n2/n1, phi/rt)",
         dict(with_proxy=True, with_depth=False, with_rank=False)),
        ("G2 井深归一化位置 1 列 (depth_norm)",
         dict(with_proxy=False, with_depth=True, with_rank=False)),
        ("G3 井内秩百分位 6 列 (rank_DEN/AC/CNL/GR/SP/RT)",
         dict(with_proxy=False, with_depth=False, with_rank=True)),
        ("G4 全部追加",
         dict(with_proxy=True, with_depth=True, with_rank=True)),
    ]
    print("\n" + "=" * 96)
    print("变体（唯一变量 = 追加的特征列；分类器与回归器一起重拟合）")
    print("=" * 96)
    rows = []
    for name, kw in GROUPS:
        t = time.time()
        Xtr = add_proxy(make_features(tr, context=True, priors=True), tr, **kw)
        Xva = add_proxy(make_features(va, context=True, priors=True), va, **kw)
        got = fit_two_stage(Xtr, Ytr)
        tv, dv, _, _ = score({c: got[c]["clf"] for c in TARGETS},
                             {c: got[c]["reg"] for c in TARGETS},
                             Xva, ens, pcn, Yva, Dva)
        d = tv - t_base
        print(f"\n{name}\n  特征 {Xva0.shape[1]} → {Xva.shape[1]}"
              f"   Total {t_base:.3f} → {tv:.3f}   Δ {d:+.3f} 分"
              f"{'   ← ≥+0.10' if d >= 0.10 else ''}   ({time.time() - t:.0f}s)")
        for c in TARGETS:
            print(f"     {c}  acc {d_base[c]['acc']:.4f}→{dv[c]['acc']:.4f}"
                  f"   real {d_base[c]['real']:.4f}→{dv[c]['real']:.4f}"
                  f"   def {d_base[c]['defv']:.4f}→{dv[c]['defv']:.4f}"
                  f"   Δ {(dv[c]['acc'] - d_base[c]['acc']) * 100 * WEIGHTS[c]:+.4f} 分")
        # 逐目标阈值重扫：判断增益是否依赖「更差模型 × 更大阈值」的联合配置
        best, th = tv, dict(CFG["thresholds"])
        for c in TARGETS:
            for g in GRID:
                th2 = dict(CFG["thresholds"])
                th2[c] = g
                b2, _, _, _ = score({c2: got[c2]["clf"] for c2 in TARGETS},
                                    {c2: got[c2]["reg"] for c2 in TARGETS},
                                    Xva, ens, pcn, Yva, Dva, th2)
                if b2 > best:
                    best, th = b2, th2
        print(f"     重扫最优 {best:.3f}   Δ {best - t_base:+.3f}   "
              f"阈值 {{{' '.join(f'{c}={th[c]:.3f}' for c in TARGETS)}}}"
              f"   （重扫只多 {best - tv:+.3f}）")
        rows.append((name, Xva.shape[1], tv, d, best))
        del got, Xtr
        gc.collect()

    print("\n" + "=" * 96)
    print("排名")
    print("=" * 96)
    print(f"  {'变体':<58}{'特征':>6}{'v9 阈值':>10}{'Δv9':>9}{'重扫最优':>10}{'Δopt':>9}")
    print(f"  {'基线':<58}{Xva0.shape[1]:>6}{t_base:>10.3f}{'':>9}{t_base:>10.3f}{'':>9}")
    for n, nf, tv, d, bo in sorted(rows, key=lambda x: -x[3]):
        print(f"  {n:<58}{nf:>6}{tv:>10.3f}{d:>+9.3f}{bo:>10.3f}{bo - t_base:>+9.3f}")

    print(f"\n总用时 {time.time() - t0:.0f}s   峰值 {rss()}")


if __name__ == "__main__":
    main()
