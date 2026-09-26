"""核查：联合标签 AUC 0.9743 是真信号，还是掩码假象？

P2（tunex_joint_label.py）报 all_def 标签 AUC = 0.9743，远高于 POR 单目标 0.8924，
号称是本项目历史上最高的分段 AUC。但两者标签**几乎完全相同**：
  P(POR_def | M.sum=2) = 0.0067，P(POR_def | M.sum=3) = 1.0000
→ POR 默认 ⟺ 三目标全默认（混合段占比仅 4.5%，其中 POR 参与率 <1%）。

**两个几乎相同的标签不可能有 0.8924 vs 0.9743 的 AUC 差距** —— 除非两者算在
**不同的行掩码**上。P2 自己标注了这一点：
  验证集 SW 少 10,288 行 = 病态井 c7611b0148bb（SW 全 NaN）
  三目标均非 NaN：验证 62,491 / 72,779（剔除 10,288）
"三目标均非 NaN" 掩码正好**把病态井整口剔除**，而那口井几乎 100% 是真实段
（POR 默认占比 0.001），是最可能拖垮 AUC 的井。

本脚本用**纯缓存**（不 fit、不 import torch）做判决性检查：
用现成的 POR p_def 去对 all_def 标签算 AUC，在两个掩码下对比。
如果剔除病态井后 AUC 跳到 ~0.97，就证明 0.9743 是掩码假象而非新信息。

用法: python experiments/diag_jointmask.py
"""
from __future__ import annotations
import os, sys, gc, time, warnings
warnings.filterwarnings("ignore")
import joblib, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sklearn.metrics import roc_auc_score
from src.feature import load_wells, make_targets, default_mask
from src.utils import TARGETS

RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
PATHO = "c7611b0148bb"   # 病态井：POR 默认占比 0.001、SW 全 NaN


def main() -> None:
    t0 = time.time()
    wells = load_wells("./train", has_labels=True)
    ids = [w["logId"].iloc[0] for w in wells]
    va = pd.concat([w for w in wells if w["logId"].iloc[0] in set(ids[72:])], ignore_index=True)
    del wells; gc.collect()
    va = va.sort_index().reset_index(drop=True)
    Y = make_targets(va)
    DM = default_mask(Y)
    well = va["logId"].to_numpy()

    cache = joblib.load(os.path.join(RUNS, "screen_tree_cache.joblib"))
    pt = cache["p_def"]
    n = len(va)
    assert n == 72779

    # 三个标签口径
    M = DM.to_numpy()                              # (n,3) True = 默认段
    all_def = M.sum(axis=1) == 3                   # 三目标全默认
    any_def = M.sum(axis=1) == 0                   # 三目标全非默认
    lab = {c: DM[c].to_numpy() for c in TARGETS}

    # 两种掩码
    m_poro = np.isfinite(Y["POR"].to_numpy())      # POR 非 NaN（单目标口径）
    m_all3 = np.isfinite(Y["POR"].to_numpy()) & np.isfinite(Y["PERM"].to_numpy()) \
        & np.isfinite(Y["SW"].to_numpy())          # 三目标全非 NaN（P2 的联合口径）

    print("=" * 78)
    print(f"验证集 {n} 行")
    print(f"  m_poro  (POR 非 NaN)        : {m_poro.sum():>7} 行")
    print(f"  m_all3  (三目标全非 NaN)    : {m_all3.sum():>7} 行  (P2 联合标签口径)")
    print(f"  差集 = {int(m_poro.sum() - m_all3.sum())} 行"
          f"  ← {'整口病态井' if m_poro.sum() - m_all3.sum() > 9000 else '少量行'}")
    for c in TARGETS:
        print(f"     {c} NaN 行数: {int((~np.isfinite(Y[c].to_numpy())).sum())}")
    print()

    # 标签相同性：all_def 与 POR_default 的逐点一致率
    agr = np.mean(lab["POR"] == all_def)
    print(f"[标签一致率] POR_default == all_def : {agr*100:.4f}%"
          f"   不一致点 {int((lab['POR'] != all_def).sum())} 个")
    print(f"               all_def 占比 {all_def.mean():.4f}   "
          f"POR_default 占比 {lab['POR'].mean():.4f}")
    print()

    # 核心检查：同一份 p_def，换标签和换掩码，AUC 各是多少
    print("=" * 78)
    print("[核心] 用缓存的树 p_def 算 AUC —— 只变标签、只变掩码")
    hdr = f"{'预测分数':<24s}"
    for maskname, mask in (("m_poro 单目标口径", m_poro), ("m_all3 三目标口径", m_all3)):
        hdr += f" | {maskname:<26s}"
    print(hdr)
    print("-" * 108)

    for pkey, p in pt.items():
        row = f"p_def_{pkey:<18s}"
        for maskname, mask in (("单目标", m_poro), ("三目标", m_all3)):
            yb = all_def[mask]
            if len(np.unique(yb)) < 2:
                row += f" | {maskname:<26s}"
                continue
            row += f" | AUC(all_def)={roc_auc_score(yb, p[mask]):.4f}"
        print(row)

    print("-" * 108)
    print("  对照：同一 p_def，用**本目标**标签")
    for pkey, p in pt.items():
        c = pkey
        row = f"p_def_{pkey:<18s}"
        for maskname, mask in (("单目标", m_poro), ("三目标", m_all3)):
            yb = lab[c][mask]
            row += f" | AUC({c})={roc_auc_score(yb, p[mask]):.4f}"
        print(row)

    print("=" * 78)
    print("[结论判据]")
    print("  若 AUC(all_def) 在 m_all3 下明显高于 m_poro 下（且两者标签一致率 99.97%），")
    print("  则 0.9743 是**剔除病态井的掩码效应**，不是新信息；")
    print("  若两掩码下接近，则联合标签确实自带信息。")
    print(f"  用时 {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
