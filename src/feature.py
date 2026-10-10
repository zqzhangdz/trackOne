"""数据读取、哨兵值处理与特征工程（点式 + 上下文 + 物理先验，单一特征源）."""

from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

from .utils import (
    CNL_HIGH,
    CNL_LOW,
    EPS,
    FEATURES,
    RAW_COLS,
    SENTINEL_VALUES,
    TARGETS,
    TRAIN_COLS,
)

# 参与上下文特征的曲线（DEPTH 只做 lag 无意义，单独处理）
CURVE_COLS = ["GR", "PE", "SP", "CAL", "AC", "DEN", "CNL", "RXO", "RT", "DEVI", "AZIM", "BIT"]
# v9：上下文回到 v7 的 ±10 / w21（225 特征），另加 12 列逐井 z 分数（{c}_wz，见
# _context_for_well 末尾）→ 237 特征。
#
# v8 把窗口放宽到 ±60 / w41（321 特征）在 harness 上 +0.065（73.760 → 73.825），但
# 225 + 逐井 z 分数在**同一 harness、同一重拟合**下是 73.931（+0.171 vs 73.760），
# 321 + 逐井 z 分数只有 73.892 —— 宽上下文被证明是负资产：±60 m 内高度冗余的局部细节
# 稀释了电平偏移这条真正的新信号。留一井包络 [+0.126, +0.183] 不跨 0、7/8 井为正。
# 依据见 experiments/runs/wellnorm.json、wellnorm_v8.log 与 CLAUDE.md §4.14。
LAG_OFFSETS = [-10, -5, -3, -2, -1, 1, 2, 3, 5, 10]
WIN_SIZES = [5, 11, 21]


def _is_sentinel(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    bad = s.isna()
    for v in SENTINEL_VALUES:
        bad = bad | np.isclose(s, v)
    bad = bad | (s.abs() > 1e5)
    return bad


def load_wells(data_dir: str, has_labels: bool) -> list[pd.DataFrame]:
    files = sorted(glob.glob(os.path.join(data_dir, "*.txt")))
    if not files:
        raise FileNotFoundError(f"目录 {data_dir} 下没有找到 .txt 数据文件")
    cols = TRAIN_COLS if has_labels else RAW_COLS
    wells = []
    for f in files:
        df = pd.read_csv(f, skiprows=2, names=cols, dtype=float)
        df["logId"] = os.path.splitext(os.path.basename(f))[0]
        wells.append(df)
    return wells


def add_physics_priors(X: pd.DataFrame) -> pd.DataFrame:
    """测井物理先验特征（仅依赖输入曲线，推理可用）."""
    X = X.copy()
    X["POR_D"] = (2.65 - X["DEN"]) / 1.65 * 100.0
    X["POR_S"] = (X["AC"] - 55.5) / (189.0 - 55.5) * 100.0
    X["POR_avg"] = (X["POR_D"] + X["CNL"]) / 2.0
    X["POR_ds"] = X["POR_D"] - X["POR_S"]
    return X


def _wellz_dict(g: pd.DataFrame) -> dict[str, np.ndarray]:
    """12 列逐井 z 分数（wellz_*）。统计量只用本井数据，推理可用，无标签泄漏。"""
    out = {}
    for c in CURVE_COLS:
        s = g[c]
        mu = s.mean()
        sd = s.std(ddof=0)                      # ddof=0 对齐 np.nanstd
        sd = float(sd) if (pd.notna(sd) and sd > 0) else 1.0
        out[f"wellz_{c}"] = ((s - mu) / sd).to_numpy()
    return out


def _context_for_well(g: pd.DataFrame, wellz: bool = True) -> pd.DataFrame:
    """单井上下文特征：lag/lead、窗口均值/标准差、一阶差分（按深度排序计算）."""
    g = g.sort_values("DEPTH")
    cols: dict[str, np.ndarray] = {
        "rel_depth": g["DEPTH"].to_numpy() - g["DEPTH"].to_numpy()[0]
    }
    for c in CURVE_COLS:
        s = g[c]
        for k in LAG_OFFSETS:
            cols[f"{c}_lag{k}"] = s.shift(k).to_numpy()
        for w in WIN_SIZES:
            r = s.rolling(w, center=True, min_periods=1)
            cols[f"{c}_wmean{w}"] = r.mean().to_numpy()
            cols[f"{c}_wstd{w}"] = r.std().fillna(0.0).to_numpy()
        cols[f"{c}_diff1"] = s.diff(1).to_numpy()
    # 逐井 z 分数（12 列，追加在末尾）：本井电平偏移。每口井的电平基线本身就是
    # 储层/井型信息 —— 所以这里是「追加」而不是「替换」原始曲线：替换会把绝对刻度
    # 抹掉，实测 POR AUC 0.8966 → 0.8619。见 experiments/tunex_wellnorm.py 与 CLAUDE.md §4.14。
    if wellz:
        cols.update(_wellz_dict(g))
    return pd.DataFrame(cols, index=g.index)


def make_features(df: pd.DataFrame, context: bool = False, priors: bool = False,
                  wellz: bool = True) -> pd.DataFrame:
    X = df[FEATURES].copy()
    for c in FEATURES:
        X.loc[_is_sentinel(X[c]), c] = np.nan
    X["CNL"] = X["CNL"].clip(CNL_LOW, CNL_HIGH)
    X["logRT"] = np.log10(X["RT"].clip(lower=EPS))
    X["logRXO"] = np.log10(X["RXO"].clip(lower=EPS))
    X["RT_over_RXO"] = X["RT"] / X["RXO"].clip(lower=EPS)
    if priors:
        X = add_physics_priors(X)
    if context:
        parts = []
        for _, g in X.groupby(df["logId"].to_numpy(), sort=False):
            parts.append(_context_for_well(g, wellz=wellz))
        ctx = pd.concat(parts)
        X = X.join(ctx)
    return X.astype(np.float32)


def make_tcn_features(df: pd.DataFrame, with_wellz: bool = False) -> pd.DataFrame:
    """TCN 通路特征：点式曲线 + 派生电阻率 + 物理先验（20 列）[+ 12 列 wellz → 32 列]。

    树通路走 context=True（225 上下文 + 12 wellz = 237），TCN 通路独立于此：
    迭代 2 的实验变量 = TCN 从未见过 wellz（老成员 20 列）。with_wellz=False
    时与旧版 `make_features(df, context=False, priors=True)` 逐点等价。
    """
    X = make_features(df, context=False, priors=True)
    if with_wellz:
        parts = []
        for _, g in X.groupby(df["logId"].to_numpy(), sort=False):
            parts.append(pd.DataFrame(_wellz_dict(g), index=g.index))
        X = X.join(pd.concat(parts))
    return X


def make_targets(df: pd.DataFrame) -> pd.DataFrame:
    Y = df[TARGETS].copy()
    for c in TARGETS:
        Y.loc[_is_sentinel(Y[c]), c] = np.nan
    return Y


def default_mask(Y: pd.DataFrame) -> pd.DataFrame:
    from .utils import DEFAULT_VALUES

    m = pd.DataFrame(index=Y.index)
    for c in TARGETS:
        m[c] = Y[c].notna() & np.isclose(Y[c], DEFAULT_VALUES[c])
    return m


def transform_target(c: str, y: pd.Series, mode: str = "perm_log") -> pd.Series:
    if mode == "all_log":
        return np.log10(y.clip(lower=EPS))
    if c == "PERM":
        return np.log10(y.clip(lower=EPS))
    return y


def inverse_transform_target(c: str, y: pd.Series, mode: str = "perm_log") -> pd.Series:
    if mode == "all_log":
        return np.power(10.0, y).clip(lower=EPS)
    if c == "PERM":
        return np.power(10.0, y).clip(lower=EPS)
    return y
