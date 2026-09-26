"""点级选择器（72 井训练版）：5 井拟合失败后，用全部 72 口训练井重测。

上一轮（experiments/tune_pointwise_selector.py）用 5 口 screen 井拟合、3 口留出：
  留出 3 井  v6 = 80.972  选择器 = 74.134（-6.84）  oracle = 88.304
选择器**劣于** v6，且其在拟合井上的分数还低于留出井 —— 即在拟合 5 井上就是在拟合噪声。
但 5 井（约 36K 点）对 8 分类、7 特征的元分类器严重不足。

本轮做法（无 GPU、不 import torch、不重训 TCN）：
  1. 3 个 72 井训练的 TCN joblib（T15/T16a/T16b）用纯 numpy 前向，
     逐井前向、只累积输出数组；
  2. 自训一棵 72 井树（同一配方：context+priors、monotonic、默认超参），
     预测全部 80 井；
  3. 官方口径逐点 oracle → HistGB 选择器，在 72 井拟合；
  4. 在 8 口 screen 井上评估（这 8 口既未参与树训练、也未参与选择器拟合）。

内存策略：逐井构建特征 → 立即转 float32 numpy → 丢弃 DataFrame；
         特征矩阵用完即 del + gc.collect()；选择器 HistGB max_iter=100。
用法: python experiments/tune_selector_72w.py
"""
from __future__ import annotations
import gc, os, sys, warnings, time, json
warnings.filterwarnings("ignore")
import joblib, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sklearn.ensemble import HistGradientBoostingClassifier
from src.feature import default_mask, load_wells, make_features, make_targets, transform_target
from src.model import _clf, _reg, POR_MONOTONIC, numpy_forward_tcn, _sigmoid_np
from src.utils import DEFAULT_VALUES, EPS, TARGETS, WEIGHTS

RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
TRAIN_DIR = "./train"
N_VAL_WELLS = 8
TCN_FILES = ["tcn_T15_seed42_150ep.joblib", "tcn_T16_seed43_150ep.joblib",
             "tcn_T16_seed44_150ep.joblib"]
V6 = {"blend_w": {"POR": 0.54, "PERM": 0.96, "SW": 0.80},
      "thresholds": {"POR": 0.115, "PERM": 0.065, "SW": 0.040},
      "clf_tcn_w": {"POR": 0.60, "PERM": 0.75, "SW": 0.30}}
BLENDS = np.float32([0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0])   # TCN 份额；候选 0 = 默认常数


def acc_official(c, pred, true):
    p = np.asarray(pred, np.float64); t = np.asarray(true, np.float64)
    if c == "PERM":
        return np.float32(np.maximum(0.0, 1.0 - np.abs(np.log10(np.maximum(p / np.maximum(t, EPS), EPS)))))
    tol = 0.08 if c == "POR" else 0.05
    return np.float32(np.maximum(0.0, 1.0 - np.abs(p - t) / (tol * (np.abs(t) + EPS))))


def tcn_parts(wells):
    """逐井 numpy 前向 3 个 TCN，返回 {c: (reg_log10, pdf)} 各井 72+8 口拼接。"""
    jobs = [joblib.load(os.path.join(RUNS, f)) for f in TCN_FILES]
    med, std, dils = jobs[0]["medians"], jobs[0]["stds"], jobs[0]["dilations"]
    cols = list(jobs[0]["feature_cols"])
    cache_npz = os.path.join(RUNS, "tcn_all80_parts.npz")
    if os.path.exists(cache_npz):
        d = np.load(cache_npz)
        return {c: (np.float32(d[f"reg_{c}"]), np.float32(d[f"pdf_{c}"])) for c in TARGETS}
    reg = {c: [] for c in TARGETS}
    pdf = {c: [] for c in TARGETS}
    for i, w in enumerate(wells):
        X = make_features(w, context=False, priors=True)[cols].fillna(med)
        X = ((X - std.replace(0, 1.0)) / std.replace(0, 1.0)).to_numpy(dtype=np.float32)
        acc_r = {c: [] for c in TARGETS}
        acc_p = {c: [] for c in TARGETS}
        for j in range(3):
            o = numpy_forward_tcn(X, jobs[j]["sd"], dils)
            for c in TARGETS:
                acc_r[c].append(o[c][:, 1].astype(np.float32))
                acc_p[c].append(_sigmoid_np(o[c][:, 0]).astype(np.float32))
            del o
        for c in TARGETS:
            reg[c].append(np.mean(acc_r[c], axis=0))
            pdf[c].append(np.mean(acc_p[c], axis=0))
        del X
        if (i + 1) % 20 == 0:
            print(f"    TCN 前向 {i+1}/{len(wells)} 井  [{time.time()-T0:.0f}s]", flush=True)
        gc.collect()
    del jobs
    out = {c: (np.concatenate(reg[c]), np.concatenate(pdf[c])) for c in TARGETS}
    np.savez_compressed(cache_npz, **{f"{k}_{c}": out[c][i] for c in TARGETS for i, k in enumerate(("reg", "pdf"))})
    return out


def tree_parts(wells, val_ids):
    """自训一棵 72 井树，返回 (80 井的 tree_log/pdef, 训练用掩码)。"""
    tr = pd.concat([w for w in wells if w["logId"].iloc[0] not in set(val_ids)], ignore_index=True)
    va = pd.concat([w for w in wells if w["logId"].iloc[0] in set(val_ids)], ignore_index=True)
    Xtr = make_features(tr, context=True, priors=True).astype(np.float32)
    Ytr = make_targets(tr)
    isdef = default_mask(Ytr)
    Xva = make_features(va, context=True, priors=True).astype(np.float32)
    Yva = make_targets(va)
    isdef_va = default_mask(Yva)
    print(f"    树训练 {len(tr):,} 点 × {Xtr.shape[1]} 维  [{time.time()-T0:.0f}s]", flush=True)
    out = {}
    for c in TARGETS:
        v = Ytr[c].notna().to_numpy()
        d = isdef[c].to_numpy()
        clf = _clf()
        clf.fit(Xtr.loc[v], d[v].astype(int))
        mono = POR_MONOTONIC if c == "POR" else None
        real = v & ~d
        reg = _reg(monotonic=mono, columns=list(Xtr.columns))   # 第 2 个位置参数是 monotonic
        reg.fit(Xtr.loc[real], transform_target(c, Ytr.loc[real, c]).to_numpy())
        ptr = clf.predict_proba(Xtr)[:, 1].astype(np.float32)
        prv = clf.predict_proba(Xva)[:, 1].astype(np.float32)
        ltr = reg.predict(Xtr).astype(np.float32)
        lrv = reg.predict(Xva).astype(np.float32)
        # 回归器只在真实段有监督；默认段仍按 log10(clip(phys)) 给值，与部署口径一致
        def to_log(y_phys):
            return np.log10(np.maximum(y_phys, EPS)).astype(np.float32)
        out[c] = {
            # 全长（72+8 口）数组：main 里用 tr_mask 切分，故训练/验证必须拼接成一致长度
            "log_all": np.concatenate([np.where(real, ltr, to_log(Ytr[c].to_numpy())), lrv]),
            "p_all": np.concatenate([ptr, prv]),
        }
        del clf, reg, ptr, prv, ltr, lrv
        gc.collect()
        print(f"    {c} 树完成  [{time.time()-T0:.0f}s]", flush=True)
    del Xtr, Xva, Ytr, Yva, isdef, isdef_va, tr, va
    gc.collect()
    return out


T0 = time.time()


def main() -> None:
    print("载入 80 井标签 ...", flush=True)
    wells = load_wells(TRAIN_DIR, has_labels=True)
    ids = [w["logId"].iloc[0] for w in wells]
    val_ids = ids[80 - N_VAL_WELLS:]
    lens = [len(w) for w in wells]
    Y = make_targets(pd.concat(wells, ignore_index=True))

    print("TCN numpy 前向（80 井 × 3 seed）...", flush=True)
    tp = tcn_parts(wells)
    del wells
    gc.collect()

    print("训练 72 井树 ...", flush=True)
    tpart = tree_parts(load_wells(TRAIN_DIR, has_labels=True), val_ids)

    # 按井边界切出验证集
    bnd = sum(lens[:80 - N_VAL_WELLS])
    tr_mask = np.zeros(sum(lens), bool); tr_mask[:bnd] = True

    tot_v6 = tot_sel = tot_or = 0.0
    print("\n" + "=" * 98)
    print("选择器（72 井拟合）vs v6 vs 点级 oracle —— 全部在 8 口 screen 井上评估")
    print("=" * 98)
    rec = {}
    for c in TARGETS:
        gc.collect()
        tlog, tcp = tpart[c]["log_all"], tpart[c]["p_all"]
        dlog, dpdf = tp[c]
        # 全部切成「训练 72 井 / 验证 8 井」两段等长数组，后续统一用 mtr / mva 取子集
        tlog_tr, tlog_va = tlog[tr_mask], tlog[~tr_mask]
        tcp_tr, tcp_va = tcp[tr_mask], tcp[~tr_mask]
        dlog_tr, dlog_va = dlog[tr_mask], dlog[~tr_mask]
        pdf_tr, pdf_va = dpdf[tr_mask], dpdf[~tr_mask]
        pm_tr = np.float32((1 - V6["clf_tcn_w"][c]) * tcp_tr + V6["clf_tcn_w"][c] * pdf_tr)
        pm_va = np.float32((1 - V6["clf_tcn_w"][c]) * tcp_va + V6["clf_tcn_w"][c] * pdf_va)
        vlog, vcp = tlog_va, tcp_va
        ytr_all = np.float64(Y[c].to_numpy())
        ytr = ytr_all[tr_mask]; yva = ytr_all[~tr_mask]
        mtr = ~np.isnan(ytr); mva = ~np.isnan(yva)

        # 训练侧：逐候选求逐点最优（argmax）
        best_acc = np.full(int(mtr.sum()), -1.0, np.float32)
        best_idx = np.zeros(int(mtr.sum()), np.int32)

        def consider(acc, j):
            up = acc > best_acc
            np.copyto(best_acc, np.where(up, acc, best_acc))
            np.copyto(best_idx, np.where(up, j, best_idx))

        consider(acc_official(c, np.full(int(mtr.sum()), DEFAULT_VALUES[c], np.float64), ytr[mtr]), 0)
        for j, w in enumerate(BLENDS, start=1):
            consider(acc_official(c, np.power(10.0, np.float32((1 - w) * tlog_tr[mtr] + w * dlog_tr[mtr])),
                                  ytr[mtr]), j)
            gc.collect()

        feats_tr = np.float32(np.column_stack([tlog_tr[mtr], dlog_tr[mtr], pm_tr[mtr],
                                                np.abs(tlog_tr[mtr] - dlog_tr[mtr]), tlog_tr[mtr] - dlog_tr[mtr],
                                                np.abs(tlog_tr[mtr]), pm_tr[mtr] ** 2]))
        clf = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.1,
                                             max_leaf_nodes=31, random_state=42)
        clf.fit(feats_tr, best_idx)   # 两者都已按 mtr 取子集
        pick_va = clf.predict(np.float32(np.column_stack([vlog, dlog_va, pm_va,
                                                          np.abs(vlog - dlog_va), vlog - dlog_va,
                                                          np.abs(vlog), pm_va ** 2])))

        N = int(mva.sum())
        cand = np.zeros((len(BLENDS) + 1, N), np.float32)
        cand[0] = np.log10(np.maximum(DEFAULT_VALUES[c], EPS))
        for j, w in enumerate(BLENDS, start=1):
            cand[j] = np.float32((1 - w) * vlog[mva] + w * dlog_va[mva])
        pred_sel = np.power(10.0, cand[pick_va[mva], np.arange(N)])
        a_sel = float(acc_official(c, pred_sel, yva[mva]).mean())
        a_or = float(np.mean([acc_official(c, np.power(10.0, np.float32((1 - w) * vlog[mva] + w * dlog_va[mva])),
                                           yva[mva]).mean() for w in BLENDS]
                             + [acc_official(c, np.full(N, DEFAULT_VALUES[c], np.float64), yva[mva]).mean()]))

        is_def = pm_va[mva] >= V6["thresholds"][c]
        pred_v6 = np.where(is_def, DEFAULT_VALUES[c],
                           np.power(10.0, np.float32((1 - V6["blend_w"][c]) * vlog[mva] + V6["blend_w"][c] * dlog_va[mva])))
        a_v6 = float(acc_official(c, pred_v6, yva[mva]).mean())

        tot_v6 += WEIGHTS[c] * a_v6
        tot_sel += WEIGHTS[c] * a_sel
        tot_or += WEIGHTS[c] * a_or
        rec[c] = {"v6": round(a_v6, 4), "selector": round(a_sel, 4), "oracle_of_blends": round(a_or, 4)}
        print(f"  {c:5s} v6={a_v6:.4f}  选择器={a_sel:.4f} (Δ{(a_sel-a_v6)*100:+.3f}分)"
              f"  混合比oracle={a_or:.4f} (Δ{(a_or-a_v6)*100:+.3f}分)  [{time.time()-T0:.0f}s]")
        del clf, cand, pred_sel, pred_v6, feats_tr, best_acc, best_idx, tlog, vlog, dlog, tcp, pdf_tr, pdf_va
        gc.collect()

    print("\n" + "=" * 98)
    print(f"  Total  v6={tot_v6*100:.3f}  选择器={tot_sel*100:.3f} (Δ{(tot_sel-tot_v6)*100:+.3f}分)"
          f"  75 分缺口 {75.0-tot_sel*100:+.3f}")
    print("=" * 98)
    with open(os.path.join(RUNS, "selector_72w.json"), "w", encoding="utf-8") as f:
        json.dump({"v6": tot_v6, "selector": tot_sel, "oracle_of_blends": tot_or, "per_target": rec},
                  f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
