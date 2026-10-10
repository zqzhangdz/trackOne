"""井眼几何轨迹（造斜严重度）+ 跨井分型：两条此前完全未编码的维度。

为什么值得试
------------
§4.14 确立的唯一有效原则是「**追加**一个此前完全未编码的维度」有效（逐井 z 分数
225→237，harness +0.171）；把已有 13 条曲线重新切分/非线性重表达无效
（±60 上下文 −0.028；Kozeny-Carman / Archie 代理 −0.048，见 tune_feature_proxy.py，
因为它们只是 DEN/CNL/RT 的单调重表达，HistGB 内部本来就能拟合）。

所以候选必须是一个**结构上不存在**的量。清点后发现两条：

  H1 井眼轨迹 / 造斜严重度 DSL
      DEVI（井斜角）与 AZIM（方位角）都已在特征里，但**从未被联合起来算过**
      任何「井眼转向」量。标准 API RP 13D 的 dog-leg severity
        cos Δ = cos I1 cos I2 + sin I1 sin I2 cos(A2 − A1)
      同时用到 DEVI 与 AZIM（含方位角回绕），HistGB 无法从两条独立曲线里
      免费得到它 —— 树只能学逐列阈值切分，学不出跨列的旋转几何。
      实测分布也支持它：DEVI 全训练集最大 26.9°，测试集最大 3.37°，
      这是三条曲线里 train/test 分布差异最大的一条。

  H2 跨井分型（well typing）
      wellz = (x − mean_well) / std_well 回答「这个点在**本井**的分布中的位置」；
      rank（G3）是它的稳健版。两者都不回答「**这口井**在 80 口井中的位置」。
      对每条曲线的本井中位数再做一次跨井 z 分数，就是一个此前完全没编码的
      维度：同一 DEN 值在「高 DEN 井」和「低 DEN 井」里含义不同。
      部署可行性：只需把 80 井的「每井中位数分布的 mean/std」作为常数写死。

口径纪律：唯一变量 = 追加的特征列；特征变了**分类器与回归器一起重拟合**；TCN 走
context=False 不受影响、零 GPU；v9 阈值为主口径，另逐目标重扫判断是否联合过拟合。

用法: python experiments/tune_wellbore_crosswell.py
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
    load_wells, make_targets, default_mask, make_features, CURVE_COLS,
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


def dsl_vector(devi, azim, dep):
    """API RP 13D 造斜严重度（度 / 30 m），逐点对前一样本。

    NaN 段不跨越：任一侧无效则该点为 NaN。方位角差按 360° 回绕归一化。
    """
    r = np.full(len(dep), np.nan)
    idr = np.deg2rad(devi)
    dr = np.deg2rad(azim)
    i1, i2 = idr[:-1], idr[1:]
    a1, a2 = dr[:-1], dr[1:]
    good = np.isfinite(i1) & np.isfinite(i2) & np.isfinite(a1) & np.isfinite(a2)
    dd = np.abs(np.diff(dep))
    good &= dd > 0
    if good.any():
        c = (np.cos(i1) * np.cos(i2) + np.sin(i1) * np.sin(i2) * np.cos(a2 - a1))
        dang = np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))
        # 度 / 30 m 归一化（行业标准单位），并单独给度/米
        seg = dang / dd * 30.0
        for k in np.where(good)[0]:
            r[k + 1] = seg[k]
        r2 = dang / dd
        for k in np.where(good)[0]:
            r[k + 1] = r2[k]          # 主输出：度/米，量级 O(1) 更适合树
    return r


def add_wellbore(X: pd.DataFrame, df: pd.DataFrame, gid: np.ndarray) -> pd.DataFrame:
    """H1：井眼轨迹几何（逐井计算，避免跨井错位）。"""
    X = X.copy()
    dep = df["DEPTH"].to_numpy(dtype=float)
    devi = df["DEVI"].to_numpy(dtype=float)
    azim = df["AZIM"].to_numpy(dtype=float)
    for s in ([-99999.0, -9999, -2.534573822542048e37]):
        devi = np.where(devi == s, np.nan, devi)
        azim = np.where(azim == s, np.nan, azim)
        dep = np.where(dep == s, np.nan, dep)

    dsl = np.full(len(df), np.nan)
    azturn = np.full(len(df), np.nan)
    for g in np.unique(gid):
        m = gid == g
        d, dv, az = dep[m], devi[m], azim[m]
        ok = np.isfinite(dv) & np.isfinite(az)
        d, dv, az = d[ok], dv[ok], az[ok]
        if len(d) < 3:
            continue
        seg = dsl_vector(dv, az, d)
        # 方位角回绕后的逐点转角（只依赖 AZIM 与 DEPTH）
        wrap = np.arctan2(np.sin(np.deg2rad(np.diff(az))), np.cos(np.deg2rad(np.diff(az))))
        at = np.full(len(d), np.nan)
        at[1:] = np.degrees(wrap) / np.maximum(np.diff(d), 1e-9)
        back = np.full(len(df), np.nan)
        back[m] = seg
        back_az = np.full(len(df), np.nan)
        back_az[m] = at
        dsl, azturn = back, back_az
    X["dsl"] = dsl
    X["azturn"] = azturn
    # |dsl| 的 21 点滑窗均值：把「这一段是否正在造斜」变成局部电平
    s = pd.Series(np.abs(dsl))
    X["dsl_abs_w21"] = s.rolling(21, center=True, min_periods=3).mean().to_numpy()
    X["dsl_max_w21"] = s.rolling(21, center=True, min_periods=3).max().to_numpy()
    return X


def build_welltyping(tr: pd.DataFrame, wells: list) -> dict:
    """H2：训练井的「每井中位数」跨井分布参数（部署时写死为常数）。"""
    rows = []
    for w in wells:
        d = {}
        for c in CURVE_COLS:
            v = w[c].to_numpy(dtype=float)
            v = np.where(np.abs(v) > 1e5, np.nan, v)
            d[f"wt_{c}"] = np.nanmedian(v)
        rows.append(d)
    M = pd.DataFrame(rows)
    mu, sd = {}, {}
    for c in CURVE_COLS:
        v = M[f"wt_{c}"].to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        mu[c] = float(v.mean()) if len(v) else 0.0
        sd[c] = float(v.std()) if len(v) else 1.0
        if sd[c] < 1e-12:
            sd[c] = 1.0
    # 井深跨度与中点也是「这口井的类型」
    spans, mids = [], []
    for w in wells:
        d = w["DEPTH"].to_numpy(dtype=float)
        d = d[np.isfinite(d)]
        spans.append(d.max() - d.min())
        mids.append((d.max() + d.min()) / 2.0)
    spans, mids = np.array(spans), np.array(mids)
    return dict(mu={c: mu[c] for c in CURVE_COLS}, sd={c: sd[c] for c in CURVE_COLS},
                span_mu=float(spans.mean()), span_sd=float(spans.std()) + 1e-9,
                mid_mu=float(mids.mean()), mid_sd=float(mids.std()) + 1e-9)


def add_welltyping(X: pd.DataFrame, df: pd.DataFrame, gid: np.ndarray,
                   wt: dict, with_span=True) -> pd.DataFrame:
    """把本井的每曲线中位数映射到跨井 z 分数（常数广播，本井内不变）。"""
    X = X.copy()
    for c in CURVE_COLS:
        v = df[c].to_numpy(dtype=float)
        v = np.where(np.abs(v) > 1e5, np.nan, v)
        z = np.full(len(df), np.nan)
        for g in np.unique(gid):
            m = gid == g
            med = np.nanmedian(v[m])
            if np.isfinite(med):
                z[m] = (med - wt["mu"][c]) / wt["sd"][c]
        X[f"wtz_{c}"] = z
    if with_span:
        dep = df["DEPTH"].to_numpy(dtype=float)
        dep = np.where(np.abs(dep) > 1e5, np.nan, dep)
        z1 = np.full(len(df), np.nan)
        z2 = np.full(len(df), np.nan)
        for g in np.unique(gid):
            m = gid == g
            d = dep[m]
            d = d[np.isfinite(d)]
            if len(d) > 1:
                z1[m] = (d.max() - d.min() - wt["span_mu"]) / wt["span_sd"]
                z2[m] = ((d.max() + d.min()) / 2.0 - wt["mid_mu"]) / wt["mid_sd"]
        X["wtz_span"] = z1
        X["wtz_mid"] = z2
    return X


def add_all(X: pd.DataFrame, df: pd.DataFrame, gid: np.ndarray, wt: dict,
            with_bore=True, with_typing=True) -> pd.DataFrame:
    if with_bore:
        X = add_wellbore(X, df, gid)
    if with_typing:
        X = add_welltyping(X, df, gid, wt)
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


def eval_all(clf_map, reg_map, Xva, ens, pcn):
    """跑一次 predict，返回 {c: dict(reg=, pm=)}。阈值只在这个结果上做，不再重跑模型。"""
    out = {}
    for c in TARGETS:
        rt = np.log10(np.maximum(inverse_transform_target(
            c, pd.Series(reg_map[c].predict(Xva), index=Xva.index)).to_numpy(), EPS))
        pt = clf_map[c].predict_proba(Xva)[:, 1]
        reg = (1 - CFG["blend_w"][c]) * rt + CFG["blend_w"][c] * ens[c]
        out[c] = dict(reg=np.where(np.isfinite(reg), reg, -5.0),
                      pm=(1 - CFG["clf_tcn_w"][c]) * pt + CFG["clf_tcn_w"][c] * pcn[c])
    return out


def accs_from(out, Yva, ths):
    accs = {}
    for c in TARGETS:
        y = Yva[c].to_numpy().astype(float)
        valid = np.isfinite(y)
        pred = np.where(out[c]["pm"] >= ths[c], DEFAULT_VALUES[c], np.power(10.0, out[c]["reg"]))
        accs[c] = np.where(valid, acc_vec(c, pred, y), np.nan)
    return accs


def score(accs, Yva, Dva):
    tot, parts = 0.0, {}
    for c in TARGETS:
        a = accs[c]
        y = Yva[c].to_numpy().astype(float)
        valid = np.isfinite(y)
        tot += WEIGHTS[c] * np.nanmean(a)
        d = Dva[c].to_numpy() & valid
        parts[c] = dict(acc=float(np.nanmean(a)),
                        real=float(np.nanmean(a[valid & ~d])) if (valid & ~d).sum() else float("nan"),
                        defv=float(np.nanmean(a[valid & d])) if (valid & d).sum() else float("nan"))
    return tot * 100, parts


def per_well_total(accs, gid):
    """逐井 Total（每井内逐点等权）—— 留一井包络用，8 井才是实验单元。"""
    per = {}
    for g in np.unique(gid):
        m = gid == g
        t = sum(WEIGHTS[c] * np.nanmean(accs[c][m]) for c in TARGETS)
        per[str(g)[:8]] = t * 100
    return per


def main():
    t0 = time.time()
    wells = load_wells("./train", has_labels=True)
    ids = [w["logId"].iloc[0] for w in wells]
    by = dict(zip(ids, wells))
    tr = pd.concat([by[i] for i in ids[:72]], ignore_index=True)
    va = pd.concat([by[i] for i in ids[72:80]], ignore_index=True)
    va_wells = [by[i] for i in ids[72:80]]
    tr72 = [by[i] for i in ids[:72]]
    Ytr, Yva = make_targets(tr), make_targets(va)
    Dva = default_mask(Yva)

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

    # ---------- 基线 ----------
    gid_va = va["logId"].to_numpy()
    Xva0 = make_features(va, context=True, priors=True)
    m0, _ = build_tree(tr, weight_tol=True)
    out0 = eval_all({c: m0.classifiers[c] for c in TARGETS},
                    {c: m0.regressors[c] for c in TARGETS}, Xva0, ens, pcn)
    accs0 = accs_from(out0, Yva, CFG["thresholds"])
    t_base, d_base = score(accs0, Yva, Dva)
    pw0 = per_well_total(accs0, gid_va)
    print(f"  [自检] 基线 Total {t_base:.3f}（应复现记录的 v9 screen 73.931）"
          f"   {'✓ 复现成功' if abs(t_base - 73.931) <= 0.02 else '✗ 未能复现'}")
    if abs(t_base - 73.931) > 0.02:
        print("  ✗ 基线未复现，退出")
        return
    print(f"  [自检] 基线分解 " + "  ".join(
        f"{c} acc {d_base[c]['acc']:.4f}/real {d_base[c]['real']:.4f}/def {d_base[c]['defv']:.4f}"
        for c in TARGETS))
    print(f"  基线特征数 {Xva0.shape[1]}   {rss()}")

    # ---------- 新维度的输入可用性核查 ----------
    print("\n[核查] 新列在验证井上的取值范围与缺失率（推理期同样可算）")
    tmp = add_wellbore(make_features(va, context=True, priors=True), va, gid_va)
    for c in ["dsl", "azturn", "dsl_abs_w21", "dsl_max_w21"]:
        v = tmp[c].to_numpy(dtype=float)
        f = v[np.isfinite(v)]
        if len(f):
            print(f"   {c:<12} 缺失 {np.nanmean(np.isnan(v)) * 100:6.2f}%   "
                  f"min {np.nanmin(f):+9.4f}  中位 {np.nanmedian(f):+9.4f}  max {np.nanmax(f):+9.4f}")
        else:
            print(f"   {c:<12} 全缺失   ← 口径错了")

    wt = build_welltyping(tr, tr72)
    tmp2 = add_welltyping(tmp, va, gid_va, wt)
    cols = [c for c in tmp2.columns if c.startswith(("wtz_",))]
    print(f"   跨井分型列 {len(cols)} 列，本井内取值唯一值数（应为 1）：")
    u = tmp2.loc[gid_va == gid_va[0], cols].nunique().to_numpy()
    print(f"     max 唯一值 = {u.max()}   （>1 说明不是井级常数，口径错了）")

    GROUPS = [
        ("H1 井眼轨迹 4 列 (dsl, azturn, dsl_abs_w21, dsl_max_w21)",
         dict(with_bore=True, with_typing=False)),
        ("H2 跨井分型 14 列 (wtz_* + span + mid)",
         dict(with_bore=False, with_typing=True)),
        ("H3 全部追加", dict(with_bore=True, with_typing=True)),
    ]
    print("\n" + "=" * 96)
    print("变体（唯一变量 = 追加的特征列；分类器与回归器一起重拟合）")
    print("=" * 96)
    rows = []
    for name, kw in GROUPS:
        t = time.time()
        gid_tr = tr["logId"].to_numpy()
        Xtr = add_all(make_features(tr, context=True, priors=True), tr, gid_tr, wt, **kw)
        Xva = add_all(make_features(va, context=True, priors=True), va, gid_va, wt, **kw)
        got = fit_two_stage(Xtr, Ytr)
        out = eval_all({c: got[c]["clf"] for c in TARGETS},
                       {c: got[c]["reg"] for c in TARGETS}, Xva, ens, pcn)
        accs = accs_from(out, Yva, CFG["thresholds"])
        tv, dv = score(accs, Yva, Dva)
        pw = per_well_total(accs, gid_va)
        d = tv - t_base
        print(f"\n{name}\n  特征 {Xva0.shape[1]} → {Xva.shape[1]}"
              f"   Total {t_base:.3f} → {tv:.3f}   Δ {d:+.3f} 分"
              f"{'   ← ≥+0.10' if d >= 0.10 else ''}   ({time.time() - t:.0f}s)")
        for c in TARGETS:
            print(f"     {c}  acc {d_base[c]['acc']:.4f}→{dv[c]['acc']:.4f}"
                  f"   real {d_base[c]['real']:.4f}→{dv[c]['real']:.4f}"
                  f"   def {d_base[c]['defv']:.4f}→{dv[c]['defv']:.4f}"
                  f"   Δ {(dv[c]['acc'] - d_base[c]['acc']) * 100 * WEIGHTS[c]:+.4f} 分")
        # 留一井包络（§6 晋级判据；8 井是实验单元，行级 bootstrap 无效）
        keys = list(pw0)
        dls = np.array([pw[k] - pw0[k] for k in keys])
        for k, x in zip(keys, dls):
            print(f"     {k}  {pw0[k]:.3f} → {pw[k]:.3f}   Δ {x:+.3f}")
        print(f"     包络 [{dls.min():+.3f},{dls.max():+.3f}]   正号 {int((dls > 0).sum())}/8"
              f"   {'   ← 包络不跨 0' if dls.min() > 0 else ''}")
        # 阈值重扫（纯 numpy，不再重跑模型）：判断增益是否依赖更大的阈值
        best, th = tv, dict(CFG["thresholds"])
        for c in TARGETS:
            for g in GRID:
                th2 = dict(CFG["thresholds"])
                th2[c] = g
                b2, _ = score(accs_from(out, Yva, th2), Yva, Dva)
                if b2 > best:
                    best, th = b2, th2
        print(f"     重扫最优 {best:.3f}   Δ {best - t_base:+.3f}   "
              f"阈值 {{{' '.join(f'{c}={th[c]:.3f}' for c in TARGETS)}}}"
              f"   （重扫只多 {best - tv:+.3f}）")
        rows.append((name, Xva.shape[1], tv, d, best, float(dls.min())))
        del got, Xtr
        gc.collect()

    print("\n" + "=" * 96)
    print("排名")
    print("=" * 96)
    print(f"  {'变体':<54}{'特征':>6}{'v9 阈值':>10}{'Δv9':>9}{'重扫最优':>10}{'Δopt':>9}{'包络下界':>10}")
    print(f"  {'基线':<54}{Xva0.shape[1]:>6}{t_base:>10.3f}{'':>9}{t_base:>10.3f}{'':>9}{'':>10}")
    for n, nf, tv, d, bo, lo in sorted(rows, key=lambda x: -x[3]):
        print(f"  {n:<54}{nf:>6}{tv:>10.3f}{d:>+9.3f}{bo:>10.3f}{bo - t_base:>+9.3f}{lo:>+10.3f}")

    print(f"\n总用时 {time.time() - t0:.0f}s   峰值 {rss()}")


if __name__ == "__main__":
    main()
