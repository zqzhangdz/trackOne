"""v8 落地：把逐目标 clf_tcn_w 写回 model.pkl 并做部署前核验。

为什么要单独写：`train.py --clf_tcn_w` 只接受**标量**（train.py:374-375），
而部署需要逐目标 dict {POR 0.60, PERM 0.70, SW 0.30}。
更严重的是：`src/model.py:150` 的判断是 `if cw_all is not None and p_clf_tcn[c]:`，
**模型里缺 clf_tcn_w 时不会报错，而是静默跳过树与 TCN 的概率融合** —— 模型行为
改变但没有任何提示。所以必须显式写回，并且核验。

用法: python experiments/apply_v8_config.py [--dry-run]
"""
from __future__ import annotations
import argparse
import joblib
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PKL = os.path.join(ROOT, "models", "model.pkl")

# v8 目标配置：特征窗口 ±60/w41（src/feature.py），融合沿用 v7，阈值取
# ctx_width_jackknife.log 在 72/8 切分上扫出的最优值。
CFG = {
    "blend_w": {"POR": [0.5, 1 / 3, 1 / 3, 1 / 3],
                "PERM": [0.5, 1 / 3, 1 / 3, 1 / 3],
                "SW": [0.25, 0.25, 0.25, 0.25]},
    "thresholds": {"POR": 0.08, "PERM": 0.28, "SW": 0.05},
    "clf_tcn_w": {"POR": 0.60, "PERM": 0.70, "SW": 0.30},
}
EXPECT_FEATURES = 321
EXPECT_TCN_PATHS = 4   # 1 树 + 3 seed TCN


def check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"  [{'OK ' if ok else 'FAIL'}] {name}{(' — ' + detail) if detail else ''}")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    m = joblib.load(PKL)
    print(f"读取 {PKL}\n")

    ok = True
    # ── 概率融合权重（最关键：缺失会导致静默跳过融合）
    # assemble 不会写 clf_tcn_w（--clf_tcn_w 只接受标量，默认 0.0 即不写）。
    # 所以这里必须写回；dry-run 只报告"将写回"，按写回后的值判 OK。
    cw = m.get("clf_tcn_w")
    if not args.dry_run:
        m["clf_tcn_w"] = dict(CFG["clf_tcn_w"])
    cw_eff = m.get("clf_tcn_w") if not args.dry_run else dict(CFG["clf_tcn_w"])
    ok &= check("clf_tcn_w 将存在（缺失会静默跳过树/TCN 概率融合）", True,
                f"assemble 后 {cw!r}"
                + ("  → 将写回 {0.60, 0.70, 0.30}" if cw is None else "  → 已存在"))
    ok &= check("clf_tcn_w == {0.60, 0.70, 0.30}",
                cw_eff == CFG["clf_tcn_w"], f"写回后 {cw_eff!r}")

    # ── 决策阈值
    ok &= check("thresholds == {0.08, 0.28, 0.05}",
                m.get("thresholds") == CFG["thresholds"], f"当前 {m.get('thresholds')!r}")

    # ── 回归融合权重
    bw = m.get("blend_w")
    bw_ok = isinstance(bw, dict) and all(len(bw[c]) == EXPECT_TCN_PATHS for c in CFG["blend_w"])
    ok &= check(f"blend_w 为逐目标 dict 且每目标 {EXPECT_TCN_PATHS} 路", bw_ok,
                f"当前 {bw!r}")

    # ── TCN 成员
    sds = m.get("tcn_sds")
    ok &= check("len(tcn_sds) == 3（3 个 seed）", isinstance(sds, list) and len(sds) == 3,
                f"当前 {None if sds is None else len(sds)}")
    ok &= check("无 tcn2_sd（pass2 未启用）", m.get("tcn2_sd") is None)

    # ── 特征维度（v8 的 321 列）
    fc = None
    try:
        sc = joblib.load(os.path.join(ROOT, "models", "scaler.pkl"))
        fc = sc.get("feature_columns")
    except Exception as e:  # pragma: no cover
        check("加载 scaler.pkl", False, str(e))
    ok &= check(f"scaler 特征列 == {EXPECT_FEATURES}",
                fc is not None and len(fc) == EXPECT_FEATURES,
                f"当前 {None if fc is None else len(fc)}")
    if fc is not None:
        cls = m.get("classifiers", {})
        regs = m.get("regressors", {})
        for c in ("POR", "PERM", "SW"):
            cl = cls.get(c) if isinstance(cls, dict) else None
            rg = regs.get(c) if isinstance(regs, dict) else None
            ns = [n for x in (cl, rg) if x is not None
                  for n in [getattr(x, "n_features_in_", None)] if n is not None]
            ok &= check(f"{c} 分类器/回归器特征数 == {len(fc)}",
                        bool(ns) and all(int(n) == len(fc) for n in ns),
                        f"当前 {ns}")

    ok &= check("target_transform == 'perm_log'",
                m.get("target_transform", "perm_log") == "perm_log",
                f"当前 {m.get('target_transform', 'perm_log')!r}")

    # ── POR 回归器的 DEN/AC 单调约束（训练时唯一施加单调约束的目标）
    por_rg = m.get("regressors", {}).get("POR")
    cst = (por_rg.get_params().get("monotonic_cst")
           if por_rg is not None else None)
    ok &= check("POR 回归器带单调约束 monotonic_cst",
                cst is not None and len(cst) == EXPECT_FEATURES,
                f"当前长度 {None if cst is None else len(cst)}")

    print()
    if not ok:
        print("有核验项失败，未写回。")
        return
    if args.dry_run:
        print("dry-run：未写回。")
        return

    joblib.dump(m, PKL)
    print(f"已写回 {PKL}")
    m2 = joblib.load(PKL)
    print(f"复核：clf_tcn_w={m2['clf_tcn_w']}  thresholds={m2['thresholds']}  "
          f"len(tcn_sds)={len(m2['tcn_sds'])}  无 tcn2_sd={m2.get('tcn2_sd') is None}")


if __name__ == "__main__":
    main()
