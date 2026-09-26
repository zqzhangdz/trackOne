"""PERM 点级模型（可部署，不需真值分段）能到多少？

前置结论（tune_perm_level.py，真值分段上界）：
  纯测井段级最佳 PERM real = 0.4492（+0.657 分）
  纯测井段级 + 真值 POR     = 0.4963（+1.307 分，上界，POR 推理时不可得）
  纯测井段级 + 树预测 POR   = 0.4492（+0.657 分）
  真值段均值 oracle         = 0.8429
  75 分线需要 PERM real ≥ 0.5241

段级模型要段边界，推理时只能靠分类器近似。本脚本改测**完全不需要分段**的点级 HistGB：
  - 只在真实段上训练（与部署口径一致：分类器判默认段就走常数）
  - 损失换成 log10 空间的 MAE —— 官方 PERM 指标正是 log10-MAE，
    而 HistGB 默认 squared_error 预测均值，不是指标最优解
  - 追加「树预测 POR」做级联特征（PERM 物理上是 POR + 岩性的函数）
  - 试段级/点级混合

评估：screen 8 井真实段逐点等权，官方 PERM 口径。
内存：只训 PERM 一个目标；X 是 float32，用完即 del。峰值约 1.5 GB。
用法: python experiments/tune_perm_deployable.py
"""
from __future__ import annotations
import gc, os, sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.feature import default_mask, load_wells, make_features, make_targets, transform_target
from src.model import _reg, POR_MONOTONIC
from src.utils import EPS, WEIGHTS

# v6 实测（screen 8 井，官方口径）
V6 = {"POR": 0.1694, "PERM": 0.4015, "SW": 0.1450}
F_DEF = 0.6063        # PERM 默认段占比（screen 8 井）
ACC_DEF = 0.9941      # PERM 默认段 acc（v6 实测）
NEED = 0.5241         # 单靠 PERM 够 75 所需的 real acc


def acc_perm_official(pred, true):
    p = np.asarray(pred, np.float64); t = np.asarray(true, np.float64)
    return float(np.mean(np.maximum(0.0, 1.0 - np.abs(np.log10(np.maximum(p / np.maximum(t, EPS), EPS))))))


POR_ACC_FULL = 0.6609   # v6 实测 POR 总 acc（screen 8 井）
SW_ACC_FULL = 0.7673    # v6 实测 SW 总 acc（screen 8 井）


def perm_total(a_perm_real):
    """PERM real acc → 总分（POR/SW 维持 v6 实测，acc_def / f_def 不变）。"""
    a_perm = F_DEF * ACC_DEF + (1 - F_DEF) * a_perm_real
    return float((a_perm * WEIGHTS["PERM"] + WEIGHTS["POR"] * POR_ACC_FULL
                  + WEIGHTS["SW"] * SW_ACC_FULL) * 100)


def main() -> None:
    t0 = time.time()
    wells = load_wells("./train", has_labels=True)
    ids = [w["logId"].iloc[0] for w in wells]
    tr_ids, va_ids = set(ids[:72]), set(ids[72:])
    tr_df = pd.concat([w for w in wells if w["logId"].iloc[0] in tr_ids], ignore_index=True)
    va_df = pd.concat([w for w in wells if w["logId"].iloc[0] in va_ids], ignore_index=True)
    del wells; gc.collect()

    trY, vaY = make_targets(tr_df), make_targets(va_df)
    trDM, vaDM = default_mask(trY), default_mask(vaY)

    Xtr = make_features(tr_df, context=True, priors=True).astype(np.float32)
    Xva = make_features(va_df, context=True, priors=True).astype(np.float32)
    print(f"特征 {Xtr.shape[0]:,} 训练点 × {Xtr.shape[1]} 维  [{time.time()-t0:.0f}s]", flush=True)

    # ── 级联特征：树预测 POR（推理时可得）
    por_ok = trY["POR"].notna().to_numpy() & ~trDM["POR"].to_numpy()
    por_pred_tr = np.log10(np.maximum(trY["POR"].to_numpy(), EPS)).astype(np.float32)
    por_pred_va = np.log10(np.maximum(vaY["POR"].to_numpy(), EPS)).astype(np.float32)
    por_reg = _reg(monotonic=POR_MONOTONIC, columns=list(Xtr.columns))
    por_reg.fit(Xtr.loc[por_ok], transform_target("POR", trY.loc[por_ok, "POR"]).to_numpy())
    por_pred_tr[por_ok] = por_reg.predict(Xtr.loc[por_ok]).astype(np.float32)
    por_pred_va = por_reg.predict(Xva).astype(np.float32)
    del por_reg; gc.collect()
    print(f"    POR 级联特征就绪  [{time.time()-t0:.0f}s]", flush=True)

    # ── PERM 真实段
    ytr_log = np.log10(np.maximum(trY["PERM"].to_numpy(), EPS))
    yva_log = np.log10(np.maximum(vaY["PERM"].to_numpy(), EPS))
    real_tr = trY["PERM"].notna().to_numpy() & ~trDM["PERM"].to_numpy()
    real_va = vaY["PERM"].notna().to_numpy() & ~vaDM["PERM"].to_numpy()
    print(f"    PERM 真实段：训练 {real_tr.sum():,} / 验证 {real_va.sum():,} 点")

    FSET = [("225 维上下文", Xtr, Xva, None, None),
            ("+预测POR（级联）", Xtr, Xva, por_pred_tr, por_pred_va)]
    LOSSES = [("squared_error", dict(loss="squared_error")),
              ("absolute_error", dict(loss="absolute_error")),
              ("quantile(0.5)", dict(loss="quantile", quantile=0.5)),
              ("quantile(0.35)", dict(loss="quantile", quantile=0.35))]
    HYP = [("i400_lr0.08_leaf31_msl50", dict(max_iter=400, learning_rate=0.08,
                                             max_leaf_nodes=31, min_samples_leaf=50)),
           ("i600_lr0.06_leaf63_msl30", dict(max_iter=600, learning_rate=0.06,
                                             max_leaf_nodes=63, min_samples_leaf=30)),
           ("i1000_lr0.04_leaf63_msl30", dict(max_iter=1000, learning_rate=0.04,
                                              max_leaf_nodes=63, min_samples_leaf=30))]

    print(f"\n{'特征集':22s} {'loss':16s} {'超参':30s} {'PERM real':>10s} "
          f"{'总分':>7s} {'≥75':>4s}")
    print("-" * 100)
    best = None
    preds = {}
    for fname, fxtr, fxva, ptr, pva in FSET:
        for lname, lkw in LOSSES:
            for hname, hk in HYP:
                cols = list(fxtr.columns)
                Xtr_f = fxtr.loc[real_tr].to_numpy()
                Xva_f = fxva.loc[real_va].to_numpy()
                if ptr is not None:
                    Xtr_f = np.hstack([Xtr_f, np.atleast_2d(ptr[real_tr]).T])
                    Xva_f = np.hstack([Xva_f, np.atleast_2d(pva[real_va]).T])
                m = HistGradientBoostingRegressor(random_state=0, **lkw, **hk)
                m.fit(Xtr_f, ytr_log[real_tr])
                pred_va = m.predict(Xva_f)
                a = acc_perm_official(10.0 ** pred_va, vaY["PERM"].to_numpy()[real_va])
                tot = perm_total(a)
                preds[(fname, lname, hname)] = pred_va
                print(f"{fname:22s} {lname:16s} {hname:30s} {a:10.4f} {tot:7.3f} "
                      f"{'✓' if a >= NEED else '':>4s}")
                if best is None or a > best[0]:
                    best = (a, fname, lname, hname)
                del m, Xtr_f, Xva_f; gc.collect()

    print("-" * 100)
    print(f"最佳: {best[1]} + {best[2]} + {best[3]}  → PERM real {best[0]:.4f}"
          f"  总分 {perm_total(best[0]):.3f}")
    print(f"对比: v6 PERM real {V6['PERM']:.4f} → 总分 {perm_total(V6['PERM']):.3f}"
          f"（75 分线需 PERM real ≥ {NEED:.4f}）")


if __name__ == "__main__":
    main()
