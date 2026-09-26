"""分段分类器重训实验：官方口径下直接优化分段器（纯 sklearn，不跑 TCN）。

为什么做这个（见 CLAUDE.md §4.2 / HANDOFF_V6.md §4）：
  v7 现部署        = 73.749
  真值分段+同回归器 = 76.308   ← 分段器可追 +2.559 分
  回归器已到顶（POR/SW 已触及真值段均值 oracle），唯一杠杆是分段器。
  树的 AUC 0.8958 / PERM 0.8915 / SW 0.9727；树 AUC 与 TCN 几乎相同（0.8966 vs 0.8935），
  融合两路提不了 AUC —— 必须改分类器本身。

做法：72 井训练 / 8 井验证，**回归器（树+TCN 融合、blend_w、clf_tcn_w）与 v7 完全一致**，
只换分类器，逐候选报告 AUC 与验证集最优阈值下的官方 acc，从而隔离出「分类器本身能追多少分」。

候选（每个目标 6 个）：
  bal      class_weight="balanced"（现部署基线，0 额外拟合）
  none     class_weight=None
  mw       官方指标「判错代价」加权：w = |acc(判默认) − acc(判真实)|
  mw05     同上取 0.5 次幂（压低极端点的支配力）
  cross    + 另外两个目标的 p_def 作特征（三目标共享「是否储层段」信号）
  cross+reg + 本目标的 TCN 回归与 p_def 作特征（树看不到的序列信息）

关于 mw：PERM 容差恒 1.0 decade 很宽，真实段里靠近默认值的点误判代价远小于 1
（y=0.05 时误判成默认仍有 0.301 分），而那些点正是分类器最没把握的点。
把权重放到真正昂贵的点上，是让分类器「按官方账本」而不是按「点数」去学。

关于 class_weight / 样本加权能否提升 AUC：它们**不会**做单调重标定（阈值是事后调的），
但它们改变树的分裂分配，所以**可能**改变 AUC。本脚本同时报 AUC 与阈值最优 acc 两个量，
不把「AUC 没动」误读成「没收益」。

cross 特征的可部署性：推理时另外两个目标的分类器与 TCN 都可用，所以
「树 clf ← 其他目标 p_def / 本目标 TCN 输出」是合法的（推理顺序变成 TCN 前向 → 树 clf → 决策）。
训练侧用**同折内**预测作特征（树对特征噪声稳健），但这对 PERM 额外做了一次 2 折 OOF 复核，
防止把乐观值当收益。

内存策略（16 GB）：特征一次构建转 float32 后即丢弃 DataFrame；
每个候选 fit 完立即 del + gc.collect()；max_iter ≤ 300。
sklearn-only、不 import torch，无需子进程隔离。

用法: python experiments/tune_clf_metric.py [--targets PERM] [--keep]
  --keep 保留中间缓存（默认每次重算，约 1–2 min）
"""
from __future__ import annotations
import argparse, gc, json, os, sys, time, warnings
warnings.filterwarnings("ignore")
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.feature import (default_mask, load_wells,
                         make_features, make_targets, transform_target)
from src.model import POR_MONOTONIC, _clf, _reg
from src.utils import DEFAULT_VALUES, EPS, TARGETS, WEIGHTS

RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
CACHE = os.path.join(RUNS, "clf_retrain_inputs.joblib")
# 缓存版本：v2 起回归器按 weight_tol 加权训练（与 train.py:278-290 部署口径同源）。
# v1 的回归器是未加权的，会在**错误的回归器**上调阈值 —— 见 2026-09-26 的踩坑记录。
CACHE_VER = 2
TRAIN_DIR = "./train"
N_TRAIN_WELLS = 72
V7 = {"blend_w": {"POR": 0.50, "PERM": 0.50, "SW": 0.75},
      "thresholds": {"POR": 0.130, "PERM": 0.270, "SW": 0.050},
      "clf_tcn_w": {"POR": 0.60, "PERM": 0.70, "SW": 0.30}}
TH_GRID = np.arange(0.01, 0.991, 0.01)
T0 = time.time()


def acc_vec(c, pred, true):
    """官方口径逐点准确率向量。PERM 恒 1.0 decade 比值容差、无分母。"""
    p = np.asarray(pred, np.float64)
    t = np.asarray(true, np.float64)
    if c == "PERM":
        return np.maximum(0.0, 1.0 - np.abs(np.log10(np.maximum(p / np.maximum(t, EPS), EPS))))
    tol = 0.08 if c == "POR" else 0.05
    return np.maximum(0.0, 1.0 - np.abs(p - t) / (tol * (np.abs(t) + EPS)))


def regret_weight(c, y, reg_phys):
    """逐点「判错的代价」= |acc(判默认) − acc(判真实)|，即分段决策在该点的分量。"""
    a_d = acc_vec(c, np.full(len(y), DEFAULT_VALUES[c]), y)
    a_r = acc_vec(c, reg_phys, y)
    w = np.abs(a_d - a_r)
    w = np.clip(w, np.percentile(w, 5), np.percentile(w, 95))
    w = np.maximum(w, 1e-3)
    return (w / w.mean()).astype(np.float32)


def to_log10(c, y_phys):
    return np.log10(np.maximum(np.asarray(y_phys, np.float64), EPS)).astype(np.float32)


def sweep_best(p_def, c, rt, ens, pdf, y, m):
    """固定 v7 的 blend_w / clf_tcn_w，只在阈值上搜索，返回 (最优 acc, 最优 th, 阈值最优贡献)。"""
    pm = ((1 - V7["clf_tcn_w"][c]) * p_def + V7["clf_tcn_w"][c] * pdf)[m]
    regb = np.power(10.0, (1 - V7["blend_w"][c]) * rt[m] + V7["blend_w"][c] * ens[m])
    fd = acc_vec(c, np.full(int(m.sum()), DEFAULT_VALUES[c]), y[m])
    fr = acc_vec(c, regb, y[m])
    order = np.argsort(pm, kind="stable")
    pos = np.searchsorted(pm[order], TH_GRID, side="left")
    Fd = np.concatenate([[0.0], np.cumsum(fd[order])])
    Fr = np.concatenate([[0.0], np.cumsum(fr[order])])
    n = len(pm)
    acc = (Fr[pos] + (Fd[n] - Fd[pos])) / n
    j = int(np.argmax(acc))
    return float(acc[j]), float(TH_GRID[j]), float(WEIGHTS[c] * acc[j] * 100)


def acc_at(p_def, c, rt, ens, pdf, y, m, th):
    pm = ((1 - V7["clf_tcn_w"][c]) * p_def + V7["clf_tcn_w"][c] * pdf)[m]
    regb = np.power(10.0, (1 - V7["blend_w"][c]) * rt[m] + V7["blend_w"][c] * ens[m])
    pred = np.where(pm >= th, DEFAULT_VALUES[c], regb)
    return float(acc_vec(c, pred, y[m]).mean())


def oof_pdef(well_ids, X_tr_all, y_tr, isdef_tr, n_folds=2):
    """2 折（按井分组）交叉预测各目标 p_def —— 给 cross 特征一个诚实的版本。

    well_ids 必须是**逐行**的井号（长度 = X_tr_all 行数），不能是逐井列表：
    后者只有几十行，和逐行的布尔掩码相与会报维度不匹配。
    """
    uid, inv = np.unique(np.asarray(well_ids), return_inverse=True)
    out = {c: np.zeros(X_tr_all.shape[0], np.float32) for c in TARGETS}
    for fi in range(n_folds):
        hold = (inv % n_folds == fi)
        for c in TARGETS:
            v = ~np.isnan(y_tr[c].to_numpy()) & ~hold
            hv = hold & ~np.isnan(y_tr[c].to_numpy())
            if not hv.any():
                continue
            clf = _clf()
            clf.fit(X_tr_all.loc[v], isdef_tr.loc[v, c].astype(int).to_numpy())
            out[c][hv] = clf.predict_proba(X_tr_all.loc[hv])[:, 1].astype(np.float32)
            del clf
            gc.collect()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default="", help="逗号分隔，空=全部")
    ap.add_argument("--keep", action="store_true", help="保留/复用 clf_retrain_inputs.joblib")
    args = ap.parse_args()
    targets = [t.strip() for t in args.targets.split(",") if t.strip()] or list(TARGETS)

    # ── 载入与切分
    print("载入 80 井 ...", flush=True)
    wells = load_wells(TRAIN_DIR, has_labels=True)
    ids = [w["logId"].iloc[0] for w in wells]
    lens = [len(w) for w in wells]
    bnd = sum(lens[:N_TRAIN_WELLS])
    df = pd.concat(wells, ignore_index=True)
    row_wells = df["logId"].to_numpy()          # 逐行井号（OOF 分组用，见 oof_pdef）
    X = make_features(df, context=True, priors=True)
    Y = make_targets(df)
    DM = default_mask(Y)
    del df
    gc.collect()
    X_tr, X_va = X.iloc[:bnd], X.iloc[bnd:]
    Y_tr, Y_va = Y.iloc[:bnd], Y.iloc[bnd:]
    D_tr, D_va = DM.iloc[:bnd], DM.iloc[bnd:]
    y_all = {c: Y[c].to_numpy().astype(np.float64) for c in TARGETS}
    del X, Y, DM
    gc.collect()
    print(f"  训练 {bnd:,} 点 × {X_tr.shape[1]} 维 / 验证 {len(X_va):,} 点  [{time.time()-T0:.0f}s]",
          flush=True)

    # ── TCN 前向缓存（3 seed 平均，全部 80 井；不 import torch）
    npz = os.path.join(RUNS, "tcn_all80_parts.npz")
    d = np.load(npz)
    ens = {c: np.float32(d[f"reg_{c}"]) for c in TARGETS}
    pdf = {c: np.float32(d[f"pdf_{c}"]) for c in TARGETS}
    assert ens["POR"].shape[0] == bnd + len(X_va), (ens["POR"].shape, bnd, len(X_va))

    def fit_regs():
        """按部署口径训练 3 个回归器（含 weight_tol 样本加权，与 train.py:278-290 一致）。

        weight_tol：w = |log10 y|（PERM）或 |y|（POR/SW），截断到 [p5,p95] 后归一化，
        只加在真实段上。漏掉它回归器就与部署工件不同源 —— 实测与部署版中位数差
        0.07~0.08 个 log10，在它之上调出的阈值对部署无效（2026-09-26 踩过的坑）。
        """
        out, X_all = {}, pd.concat([X_tr, X_va])
        for c in TARGETS:
            v = ~np.isnan(y_all[c][:bnd])
            real = v & (~D_tr[c].to_numpy().astype(bool))
            reg = _reg(monotonic=POR_MONOTONIC if c == "POR" else None, columns=list(X_tr.columns))
            yt = transform_target(c, Y_tr[c].loc[real]).to_numpy()
            yv = Y_tr[c].loc[real].to_numpy().astype(np.float64)
            w = np.abs(np.log10(np.maximum(yv, EPS))) if c == "PERM" else np.abs(yv)
            w = np.clip(w, np.percentile(w, 5), np.percentile(w, 95))
            w = (w / w.mean()).astype(np.float32)
            reg.fit(X_tr.loc[real], yt, sample_weight=w)
            r_all = np.asarray(reg.predict(X_all), np.float64)
            # 回归输出统一折到 log10，与部署口径一致：
            # PERM 回归器直接输出 log10（inverse→log10 是恒等）；POR/SW 输出物理量，需再取 log10
            out[c] = r_all.astype(np.float32) if c == "PERM" else to_log10(c, r_all)
            del reg, r_all
            gc.collect()
            print(f"  {c} 回归器(weight_tol)完成  [{time.time()-T0:.0f}s]", flush=True)
        del X_all
        gc.collect()
        return out

    def fit_clfs():
        """训练 3 个基线分类器（72 井，class_weight="balanced" = 现部署配置）。"""
        out = {}
        for c in TARGETS:
            v = ~np.isnan(y_all[c][:bnd])
            clf = _clf()
            clf.fit(X_tr.loc[v], D_tr.loc[v, c].astype(int).to_numpy())
            out[c] = clf.predict_proba(X_va)[:, 1].astype(np.float32)
            del clf
            gc.collect()
            print(f"  {c} 基线分类器完成  [{time.time()-T0:.0f}s]", flush=True)
        return out

    cache_hit = os.path.exists(CACHE) and args.keep
    ver_ok = False
    if cache_hit:
        ck = joblib.load(CACHE)
        ver_ok = ck.get("ver") == CACHE_VER
        if ver_ok:
            rt_all, p_val, oof = ck["rt_all"], ck["p_val"], ck["oof"]
        else:
            # 旧版缓存：p_val / oof 只依赖分类器，可安全复用；回归器必须按 weight_tol 重训
            p_val, oof = ck["p_val"], ck["oof"]
        del ck
        gc.collect()
        if ver_ok:
            print(f"  复用缓存 {CACHE}  [{time.time()-T0:.0f}s]", flush=True)
        else:
            print("  缓存为旧版（回归器未含 weight_tol）→ 复用 p_val/oof，重训回归器", flush=True)
    else:
        print("训练 3 个基线分类器（72 井）...", flush=True)
        p_val = fit_clfs()
        print("2 折 OOF p_def（cross 特征的诚实版本）...", flush=True)
        oof = oof_pdef(row_wells[:bnd], X_tr, Y_tr, D_tr)

    if not ver_ok:
        print("训练 3 个回归器（weight_tol，与部署同源）...", flush=True)
        rt_all = fit_regs()
        joblib.dump({"rt_all": rt_all, "p_val": p_val, "oof": oof, "ver": CACHE_VER}, CACHE)
        gc.collect()

    rt_all = {c: np.float32(np.asarray(rt_all[c])[:bnd + len(X_va)]) for c in TARGETS}
    p_val = {c: np.float32(np.asarray(p_val[c])) for c in TARGETS}
    oof = {c: np.float32(np.asarray(oof[c])) for c in TARGETS}

    # ── 混合回归估计（物理空间）：用于 mw 权重
    y_tr_all = {c: y_all[c][:bnd] for c in TARGETS}
    y_va_all = {c: y_all[c][bnd:] for c in TARGETS}
    m_tr = {c: ~np.isnan(y_tr_all[c]) for c in TARGETS}
    m_va = {c: ~np.isnan(y_va_all[c]) for c in TARGETS}
    reg_phys_tr = {}
    for c in TARGETS:
        blend = (1 - V7["blend_w"][c]) * rt_all[c][:bnd] + V7["blend_w"][c] * ens[c][:bnd]
        reg_phys_tr[c] = np.power(10.0, blend)

    # ── 基线自洽性核对：本脚本重训的树分类器与回归器应复刻 screen_tree_cache
    try:
        ck2 = joblib.load(os.path.join(RUNS, "screen_tree_cache.joblib"))
        for c in targets:
            dd = float(np.abs(p_val[c][m_va[c]] - ck2["p_def"][c][m_va[c]]).mean())
            print(f"  自洽核对 {c}: 与 screen_tree_cache 的 p_def 平均绝对差 {dd:.2e} "
                  f"(应接近 0，确认基线同源)")
        # 回归器核对：weight_tol 缺失时这里会差 1e-1 量级，是部署不同源的信号
        for c in targets:
            m = m_va[c]
            d2 = np.abs(np.asarray(rt_all[c][bnd:], np.float64)[m]
                        - np.asarray(ck2["reg"][c], np.float64)[m])
            print(f"  自洽核对 {c}: 与 screen_tree_cache 的 reg 中位绝对差 "
                  f"{np.median(d2):.2e} / 均值 {d2.mean():.2e}（应接近 0）")
    except Exception as e:
        print(f"  自洽核对跳过: {e}")

    # ── 逐候选评估
    CANDS = [
        ("bal",        None,                          "base",     None),
        ("none",       dict(class_weight=None),       "base",     None),
        ("mw",         dict(class_weight=None),       "base",     "regret"),
        ("mw05",       dict(class_weight=None),       "base",     "regret05"),
        ("cross",      None,                          "cross",    None),
        ("cross+reg",  None,                          "crossreg", None),
    ]
    rows = []
    print("\n" + "=" * 108)
    print("分段器候选（回归器/blend_w/clf_tcn_w 固定为 v7；只在分类器与阈值上变）")
    print("=" * 108)
    print(f"{'目标':6s} {'候选':10s} {'AUC':>8s} {'acc@v7th':>10s} {'acc@最优th':>11s} {'最优th':>7s} "
          f"{'Δacc×100':>10s}  说明")

    # bal 的 p_val 已在上一步算出，直接复用
    p_cache = {c: {"bal": p_val[c]} for c in TARGETS}

    for c in targets:
        others = [o for o in TARGETS if o != c]
        Xt_base, Xv_base = X_tr.to_numpy(np.float32), X_va.to_numpy(np.float32)
        feat = {
            "base": (Xt_base, Xv_base),
            # 训练侧用 2 折 OOF（不含自身），避免把同折内乐观预测当特征
            "cross": (np.concatenate([Xt_base,
                                      oof[others[0]][:, None],
                                      oof[others[1]][:, None]], axis=1),
                      np.concatenate([Xv_base,
                                      p_val[others[0]][:, None],
                                      p_val[others[1]][:, None]], axis=1)),
        }
        feat["crossreg"] = (np.concatenate([feat["cross"][0],
                                            ens[c][:bnd][:, None], pdf[c][:bnd][:, None]], axis=1),
                            np.concatenate([feat["cross"][1],
                                            ens[c][bnd:][:, None], pdf[c][bnd:][:, None]], axis=1))

        ytr, yva = y_tr_all[c], y_va_all[c]
        rt_v, ens_v, pdf_v = rt_all[c][bnd:], ens[c][bnd:], pdf[c][bnd:]

        for name, params, fset, swmode in CANDS:
            if name in p_cache[c]:
                p = p_cache[c][name]
            else:
                Xt, Xv = feat[fset]
                v = m_tr[c]
                clf = HistGradientBoostingClassifier(random_state=42, **(params or {}))
                sw, wgt = None, None
                if swmode:
                    wgt = regret_weight(c, ytr[v], reg_phys_tr[c][v])
                    sw = np.power(wgt, 0.5) if swmode == "regret05" else wgt
                clf.fit(Xt[v], D_tr[c].to_numpy().astype(int)[v].astype(np.int32),
                        sample_weight=sw)
                p = clf.predict_proba(Xv)[:, 1].astype(np.float32)
                del clf, Xt, Xv, wgt, sw
                gc.collect()
                p_cache[c][name] = p

            AUC = float(roc_auc_score(D_va[c].to_numpy().astype(int)[m_va[c]], p[m_va[c]]))
            a_v7 = acc_at(p, c, rt_v, ens_v, pdf_v, yva, m_va[c], V7["thresholds"][c])
            a_best, th_best, contrib = sweep_best(p, c, rt_v, ens_v, pdf_v, yva, m_va[c])
            note = {
                "bal": "现部署基线",
                "none": "去 class_weight",
                "mw": "官方判错代价加权",
                "mw05": "代价加权^0.5",
                "cross": "+另两目标 p_def",
                "cross+reg": "+另两目标 p_def + 本目标 TCN",
            }[name]
            rows.append({"target": c, "cand": name, "auc": AUC, "acc_v7th": a_v7,
                         "acc_best": a_best, "th_best": th_best, "contrib": contrib})
            print(f"{c:6s} {name:10s} {AUC:8.4f} {a_v7:10.4f} {a_best:11.4f} {th_best:7.3f} "
                  f"{(a_best-a_v7)*100:+10.3f}  {note}  [{time.time()-T0:.0f}s]")

    # ── 落盘候选 p_def，便于秒级复算汇总（免重跑 15 次 fit）
    joblib.dump({c: {k: np.float32(v) for k, v in d_.items()} for c, d_ in p_cache.items()},
                os.path.join(RUNS, "clf_retrain_pred.joblib"))

    # ── 汇总：每目标取最优候选
    base = {r["target"]: r for r in rows if r["cand"] == "bal"}
    print("\n" + "=" * 112)
    print("汇总：每目标取最优候选（阈值仍在验证集上调）")
    print("=" * 112)
    print(f"{'目标':6s} {'bal@v7th':>9s} {'最优候选':11s} {'acc':>9s} {'最优th':>7s} {'Δacc':>8s} "
          f"{'Δ贡献分':>9s} {'上界(真值分段)':>15s}")
    tot_bal = tot_best = tot_or = 0.0
    rowd, bestpick = {}, {}
    for c in targets:
        cand = max([r for r in rows if r["target"] == c], key=lambda r: r["acc_best"])
        bestpick[c] = cand
        b = base[c]
        regb = np.power(10.0, (1 - V7["blend_w"][c]) * rt_all[c][bnd:]
                        + V7["blend_w"][c] * ens[c][bnd:])
        m = m_va[c]
        # 上界：真值分段 + 同一回归器，在**全部有效点**上评估（只取默认段会恒为 1.0）
        dm = D_va[c].to_numpy().astype(bool)
        pred_or = np.where(dm, DEFAULT_VALUES[c], regb)
        a_or = float(acc_vec(c, pred_or[m], y_va_all[c][m]).mean())
        contrib_or = WEIGHTS[c] * a_or * 100
        tot_bal += b["contrib"]; tot_best += cand["contrib"]; tot_or += contrib_or
        # 逐行「对总分的贡献」向量，供下方留一井 jackknife 用
        # 注意 ens/pdf 是全部 80 井，必须按 bnd 切到验证 8 井再与 p_cache 组合
        pdfv, ensv = pdf[c][bnd:], ens[c][bnd:]
        pm1 = (1 - V7["clf_tcn_w"][c]) * p_cache[c]["bal"] + V7["clf_tcn_w"][c] * pdfv
        pm2 = (1 - V7["clf_tcn_w"][c]) * p_cache[c][cand["cand"]] + V7["clf_tcn_w"][c] * pdfv
        p1 = np.where(pm1 >= V7["thresholds"][c], DEFAULT_VALUES[c], regb)
        p2 = np.where(pm2 >= cand["th_best"], DEFAULT_VALUES[c], regb)
        rowd[c] = WEIGHTS[c] * (acc_vec(c, p2[m], y_va_all[c][m])
                                - acc_vec(c, p1[m], y_va_all[c][m])) / int(m.sum())
        print(f"{c:6s} {b['acc_v7th']:9.4f} {cand['cand']:11s} {cand['acc_best']:9.4f} "
              f"{cand['th_best']:7.3f} {cand['acc_best']-b['acc_v7th']:+8.4f} "
              f"{cand['contrib']-b['contrib']:+9.3f} {a_or:15.4f}")
    print("-" * 112)
    print(f"{'Total':6s} {'':>9s} {'':>11s} {'':>9s} {'':>7s} {'':>8s} {tot_best-tot_bal:+9.3f} {'':>15s}")
    print(f"\n  v7 现部署（本 harness 复算 bal，各目标阈值重调）= {tot_bal:.3f}")
    print(f"  最优候选（逐目标取最优分类器 + 各自最优阈值）  = {tot_best:.3f}  ({tot_best-tot_bal:+.3f})")
    print(f"  真值分段上界（同一回归器）                    = {tot_or:.3f}  (可追 {tot_or-tot_bal:+.3f})")
    print(f"  75 分缺口（以最优候选计）                     = {75.0-tot_best:+.3f}")

    # ── 证据强度：8 口井才是实验单元。72,779 行被 8 口井高度聚类，行级 bootstrap
    #    会严重低估方差（SE 与 1/√N 成正比），必须按井做留一 jackknife。
    vw = row_wells[bnd:]
    lids = pd.unique(vw)
    print(f"\n  逐井分解 Δ 总分（{len(lids)} 口验证井；8 井才是实验单元，"
          f"行级 bootstrap 会低估方差）")
    print(f"  {'井':34s} {'Δ总分':>8s}")
    total_d, wdelta = 0.0, {}
    for lid in lids:
        dv = 0.0
        for c in targets:
            dv += 100.0 * rowd[c][vw[np.flatnonzero(m_va[c])] == lid].sum()
        wdelta[lid] = dv
        total_d += dv
        print(f"  {lid:34s} {dv:+8.3f}")
    pos = sum(1 for v in wdelta.values() if v > 0)
    lo = total_d - max(wdelta.values())     # 去掉最大正贡献井
    hi = total_d - min(wdelta.values())     # 去掉最大负贡献井
    print(f"\n  合计 Δ = {total_d:+.3f}    正号井 {pos}/{len(lids)}")
    print(f"  留一井 jackknife [{lo:+.3f}, {hi:+.3f}]  → "
          f"{'跨 0：噪声内，不构成晋级证据' if lo <= 0 <= hi else '不跨 0：方向可信'}")
    print(f"  screen 噪声 ≈ ±0.3，本次 Δ = {total_d:+.3f} "
          f"{'在噪声带内' if abs(total_d) < 0.3 else '超出噪声带'}")

    out = os.path.join(RUNS, "clf_retrain.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"rows": rows, "total_baseline": tot_bal, "total_best": tot_best,
                   "total_oracle_seg": tot_or,
                   "delta_total": total_d,
                   "per_well_delta": wdelta,
                   "jackknife": [lo, hi],
                   "best_pick": {c: {"cand": bestpick[c]["cand"], "th": bestpick[c]["th_best"]}
                                 for c in targets}}, f, ensure_ascii=False, indent=2)
    print(f"\n  结果已写入 {out}   耗时 {time.time()-T0:.0f}s")


if __name__ == "__main__":
    main()
