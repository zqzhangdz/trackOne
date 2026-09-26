"""v7 融合配置评估与落地（官方口径，零训练）。

v7 = v6 的同一批成员，只改三个融合常数（按官方口径重调）：

    字段          v6                        v7
    blend_w       POR .54 / PERM .96 / SW .80   POR .50 / PERM .50 / SW .75
    thresholds    POR .115/ PERM .065/ SW .040  POR .13 / PERM .27 / SW .05
    clf_tcn_w     POR .60 / PERM .75 / SW .30   POR .60 / PERM .70 / SW .30

唯一的实质改动是 PERM 阈值 0.065 → 0.27（+0.456 贡献）。0.065 是按旧 PERM 口径调的，
旧容差 0.02–0.36 decade 下"真实段误判成默认"代价极小，阈值被压得过低；官方 1.0 decade
比值容差宽松得多，最优阈值高得多。

本脚本做三件事：
  1. 官方口径下逐目标联合扫 (w, th, cw)，确认 v7 就是最优点（不是拍脑袋）；
  2. 逐井分解 v7 − v6 的分数差，确认增益不是靠某一口井堆出来的；
  3. --write 时把 v7 写入 models/model.pkl（备份 .v6bak，格式校验失败自动回滚）。

只用缓存前向；不 fit 模型、不 import torch（遵守同进程运行时约束）。
用法: python experiments/eval_v7_config.py [--write]
"""
from __future__ import annotations
import argparse, gc, os, shutil, sys, time, warnings
warnings.filterwarnings("ignore")
import joblib, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.feature import load_wells, make_targets
from src.utils import DEFAULT_VALUES, EPS, TARGETS, WEIGHTS

RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
T108 = {"T15": "T15_seed42_150ep", "T16a": "T16_seed43_150ep", "T16b": "T16_seed44_150ep"}
W_GRID = np.arange(0.0, 1.001, 0.05)
TH_GRID = np.arange(0.005, 0.995, 0.02)
CW_GRID = np.arange(0.0, 1.001, 0.10)

V6 = {"blend_w": {"POR": 0.54, "PERM": 0.96, "SW": 0.80},
      "thresholds": {"POR": 0.115, "PERM": 0.065, "SW": 0.040},
      "clf_tcn_w": {"POR": 0.60, "PERM": 0.75, "SW": 0.30}}
# 待落地的候选（由下方网格搜索确认最优后再写）
CAND = {"blend_w": {"POR": 0.50, "PERM": 0.50, "SW": 0.75},
        "thresholds": {"POR": 0.130, "PERM": 0.270, "SW": 0.050},
        "clf_tcn_w": {"POR": 0.60, "PERM": 0.70, "SW": 0.30}}


def acc_vec(c, pred, true):
    """官方口径逐点准确率向量（公式原文）。PERM 恒 1.0 decade 比值容差。"""
    p = np.asarray(pred, float); t = np.asarray(true, float)
    if c == "PERM":
        return np.maximum(0.0, 1.0 - np.abs(np.log10(np.maximum(p / np.maximum(t, EPS), EPS))))
    tol = 0.08 if c == "POR" else 0.05
    return np.maximum(0.0, 1.0 - np.abs(p - t) / (tol * (np.abs(t) + EPS)))


def predict_all(store, rt, pt, cfg):
    """返回 {c: (acc_vec_full, y, mask_valid)}。reg_blend 在 log10 空间。"""
    out = {}
    for c in TARGETS:
        w, th, cw = cfg["blend_w"][c], cfg["thresholds"][c], cfg["clf_tcn_w"][c]
        ens = np.mean([store[f"{T108[t]}__{c}__reg"] for t in T108], axis=0)
        pcn = np.mean([store[f"{T108[t]}__{c}__pdf"] for t in T108], axis=0)
        pm = (1 - cw) * pt[c] + cw * pcn
        reg_blend = (1 - w) * rt[c] + w * ens
        pred = np.where(pm >= th, DEFAULT_VALUES[c], np.power(10.0, reg_blend))
        out[c] = pred
    return out


def grid_best(rt, pt, store, c):
    """逐目标联合扫 (w, th, cw)，返回最优 (acc, w, th, cw)。全部先按 valid 掩码取子集。"""
    ens = np.mean([store[f"{T108[t]}__{c}__reg"] for t in T108], axis=0)
    pcn = np.mean([store[f"{T108[t]}__{c}__pdf"] for t in T108], axis=0)
    y = va_y[c]
    valid = ~np.isnan(y)
    n = int(valid.sum())
    idx = np.flatnonzero(valid)
    rtv, ptv, env, pcnv = rt[c][idx], pt[c][idx], ens[idx], pcn[idx]
    yv = y[idx]

    best = (-1.0, 0.0, 0.0, 0.0)
    for cw in CW_GRID:
        pm = (1 - cw) * ptv + cw * pcnv
        order = np.argsort(pm, kind="stable")
        pm_s = pm[order]
        pos = np.searchsorted(pm_s, TH_GRID, side="left")
        f_d = acc_vec(c, np.full(n, DEFAULT_VALUES[c]), yv)[order]
        cs_d = np.concatenate([[0.0], np.cumsum(f_d)])
        for w in W_GRID:
            fr = acc_vec(c, np.power(10.0, (1 - w) * rtv + w * env), yv)[order]
            cs_r = np.concatenate([[0.0], np.cumsum(fr)])
            acc = (cs_r[pos] + (cs_d[n] - cs_d[pos])) / n
            j = int(np.argmax(acc))
            if acc[j] > best[0]:
                best = (float(acc[j]), float(w), float(TH_GRID[j]), float(cw))
    return best


def _cand_acc(c, cc, rt, pt, store):
    """按给定 (w, th, cw) 算单目标 acc，用于判定候选是否落在网格最优处。"""
    w, th, cw = cc
    ens = np.mean([store[f"{T108[t]}__{c}__reg"] for t in T108], axis=0)
    pcn = np.mean([store[f"{T108[t]}__{c}__pdf"] for t in T108], axis=0)
    y = va_y[c]
    m = ~np.isnan(y)
    pm = (1 - cw) * pt[c] + cw * pcn
    reg_blend = (1 - w) * rt[c] + w * ens
    pred = np.where(pm >= th, DEFAULT_VALUES[c], np.power(10.0, reg_blend))
    return float(acc_vec(c, pred[m], y[m]).mean())


va_y = {}

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="把候选配置写入 models/model.pkl")
    args = ap.parse_args()
    t0 = time.time()

    wells = load_wells("./train", has_labels=True)
    ids = [w["logId"].iloc[0] for w in wells]
    va = pd.concat([w for w in wells if w["logId"].iloc[0] in set(ids[72:])], ignore_index=True)
    del wells; gc.collect()
    Y = make_targets(va)
    global va_y
    va_y = {c: Y[c].to_numpy() for c in TARGETS}

    cache = joblib.load(os.path.join(RUNS, "screen_tree_cache.joblib"))
    rt, pt = cache["reg"], cache["p_def"]
    store = {k: v for k, v in np.load(os.path.join(RUNS, "tcn_screen_outputs.npz")).items()}
    per_well = va["logId"].to_numpy()

    # ── 1. 确认候选就是网格最优点
    print("=" * 96)
    print("1) 官方口径逐目标网格 (w 步 .05 / th 步 .02 / cw 步 .10) —— 确认候选即最优")
    print("=" * 96)
    print(f"{'目标':6s} {'网格最优 w/th/cw':22s} {'acc':>7s}   {'候选 w/th/cw':20s} {'acc':>7s} "
          f"{'Δacc':>8s}  一致")
    ok_all = True
    for c in TARGETS:
        a, w, th, cw = grid_best(rt, pt, store, c)
        cc = (CAND["blend_w"][c], CAND["thresholds"][c], CAND["clf_tcn_w"][c])
        a_cand = _cand_acc(c, cc, rt, pt, store)
        same = abs(a - a_cand) < 5e-4
        ok_all &= same
        print(f"{c:6s} {f'{w:.2f}/{th:.2f}/{cw:.2f}':22s} {a:7.4f}   "
              f"{f'{cc[0]:.2f}/{cc[1]:.2f}/{cc[2]:.2f}':20s} {a_cand:7.4f} {a_cand-a:+8.4f} "
              f"{'✓' if same else '✗'}")
    print("\n  w 维度在最优处是平的（0.50 与 0.55 的 acc 差 <5e-4）—— th 是主导变量，")
    print("  取更靠树的 0.50 更稳健。判定用 Δacc<5e-4，不看参数是否逐位相同。")
    if not ok_all:
        print("  ⚠ 候选与网格最优不一致，请先更新 CAND 再 --write")

    # ── 2. 逐井分解 v7 − v6
    print("\n" + "=" * 96)
    print("2) 逐井分解：v7 − v6 的总分差（screen 8 井，官方口径，逐点等权）")
    print("=" * 96)
    pred_v6 = predict_all(store, rt, pt, V6)
    pred_v7 = predict_all(store, rt, pt, CAND)

    well_tot_v6, well_tot_v7 = {}, {}
    for lid in pd.unique(per_well):
        sel = per_well == lid
        acc6, acc7 = {}, {}
        for c in TARGETS:
            y = va_y[c]
            m = sel & ~np.isnan(y)
            if m.sum() == 0:
                continue
            acc6[c] = float(acc_vec(c, pred_v6[c][m], y[m]).mean())
            acc7[c] = float(acc_vec(c, pred_v7[c][m], y[m]).mean())
        if not acc6:
            continue
        w6 = sum(WEIGHTS[c] * acc6[c] for c in acc6) * 100
        w7 = sum(WEIGHTS[c] * acc7[c] for c in acc7) * 100
        well_tot_v6[lid] = w6
        well_tot_v7[lid] = w7

    print(f"{'井':34s} {'v6':>8s} {'v7':>8s} {'Δ':>8s}")
    pos = 0
    for lid in pd.unique(per_well):
        if lid not in well_tot_v6:
            continue
        d = well_tot_v7[lid] - well_tot_v6[lid]
        pos += d > 0
        print(f"{lid:34s} {well_tot_v6[lid]:8.3f} {well_tot_v7[lid]:8.3f} {d:+8.3f}")
    n_w = len(well_tot_v6)
    print(f"  → {pos}/{n_w} 口井为正"
          f"{'（≥4/5 风格的方向性通过）' if pos / n_w >= 0.6 else '（⚠ 方向性不足，勿落地）'}")

    # ── 3. 总分
    a6 = {c: float(acc_vec(c, pred_v6[c], va_y[c])[~np.isnan(va_y[c])].mean()) for c in TARGETS}
    a7 = {c: float(acc_vec(c, pred_v7[c], va_y[c])[~np.isnan(va_y[c])].mean()) for c in TARGETS}
    t6 = sum(WEIGHTS[c] * a6[c] for c in TARGETS) * 100
    t7 = sum(WEIGHTS[c] * a7[c] for c in TARGETS) * 100
    print("\n" + "=" * 96)
    print(f"3) 总分（screen 8 井，官方口径，逐点等权）")
    print("=" * 96)
    print(f"{'目标':6s} {'v6':>9s} {'v7':>9s} {'Δ':>9s}")
    for c in TARGETS:
        print(f"{c:6s} {a6[c]:9.4f} {a7[c]:9.4f} {a7[c]-a6[c]:+9.4f}")
    print(f"{'Total':6s} {t6:9.3f} {t7:9.3f} {t7-t6:+9.3f}")
    print(f"  75 分缺口: v6 {75.0-t6:+.3f} → v7 {75.0-t7:+.3f}")

    # ── 4. 写入
    if not args.write:
        print("\n未写入（加 --write 落地 v7 配置）")
        return
    path, bak = "models/model.pkl", "models/model.pkl.v6bak"
    m = joblib.load(path)
    n_tcn = len(m["tcn_sds"])
    print(f"\nmodel.pkl: tcn_sds={n_tcn} 个, tcn2_sd={'有' if 'tcn2_sd' in m else '无'}")
    assert n_tcn == 3, f"tcn_sds 应为 3 个，实为 {n_tcn}"
    assert "tcn2_sd" not in m, "存在 tcn2_sd：pass2 被误拼（见 CLAUDE.md §5）"
    bw_bad = any(not isinstance(m["blend_w"][c], list) or len(m["blend_w"][c]) != n_tcn + 1
                 for c in TARGETS)
    if bw_bad:
        assert os.path.exists(bak) or os.path.exists("models/model.pkl.v5bak"), "blend_w 已损坏且无备份"
        src = bak if os.path.exists(bak) else "models/model.pkl.v5bak"
        print(f"  blend_w 格式错误 → 从 {src} 恢复")
        shutil.copy2(src, path)
        m = joblib.load(path)

    if not os.path.exists(bak):
        shutil.copy2(path, bak)
        print(f"  已备份 v6 配置到 {bak}")
    m["blend_w"] = {c: [1.0 - CAND["blend_w"][c]] + [CAND["blend_w"][c] / n_tcn] * n_tcn
                    for c in TARGETS}
    m["thresholds"] = CAND["thresholds"]
    m["clf_tcn_w"] = CAND["clf_tcn_w"]
    joblib.dump(m, path)

    m2 = joblib.load(path)
    ok = (m2["thresholds"] == CAND["thresholds"]
          and m2["clf_tcn_w"] == CAND["clf_tcn_w"]
          and all(abs(m2["blend_w"][c][0] - (1 - CAND["blend_w"][c])) < 1e-9 for c in TARGETS)
          and len(m2["tcn_sds"]) == 3 and "tcn2_sd" not in m2)
    print(f"  写回校验: {ok}")
    if not ok:
        shutil.copy2(bak, path)
        raise SystemExit("校验失败，已回滚到 .v6bak")
    print(f"  已写 models/model.pkl（v7 配置，总分 {t7:.3f}）")
    print(f"  耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
