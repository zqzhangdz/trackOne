"""数据读取与特征工程."""

from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

from .config import (
    CNL_HIGH,
    CNL_LOW,
    EPS,
    FEATURES,
    RAW_COLS,
    SENTINEL_VALUES,
    TRAIN_COLS,
    TARGETS,
)


def _is_sentinel(s: pd.Series) -> pd.Series:
    """把 NaN、哨兵值和明显异常的极大/极小值识别为缺失."""
    s = pd.to_numeric(s, errors="coerce")
    bad = s.isna()
    for v in SENTINEL_VALUES:
        bad = bad | np.isclose(s, v)
    # 任何超出正负 1e5 的值都是错误码（正常测井量级远小于此）
    bad = bad | (s.abs() > 1e5)
    return bad


def load_wells(data_dir: str, has_labels: bool) -> list[pd.DataFrame]:
    """读取目录下所有 *.txt 井文件，返回带 logId 的 DataFrame 列表."""
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


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """构造模型输入特征（train / predict 共用，保证一致）."""
    X = df[FEATURES].copy()
    for c in FEATURES:
        X.loc[_is_sentinel(X[c]), c] = np.nan

    # CNL 物理截断，抑制极端测井误差
    X["CNL"] = X["CNL"].clip(CNL_LOW, CNL_HIGH)

    # 电阻率取对数（跨数量级），并补充派生特征
    X["logRT"] = np.log10(X["RT"].clip(lower=EPS))
    X["logRXO"] = np.log10(X["RXO"].clip(lower=EPS))
    X["RT_over_RXO"] = X["RT"] / X["RXO"].clip(lower=EPS)

    return X


def make_targets(df: pd.DataFrame) -> pd.DataFrame:
    """构造目标，并把哨兵值标为 NaN（训练时掩码）."""
    Y = df[TARGETS].copy()
    for c in TARGETS:
        Y.loc[_is_sentinel(Y[c]), c] = np.nan
    return Y


def default_mask(Y: pd.DataFrame) -> pd.DataFrame:
    """逐目标标记该点是否等于默认常数（NaN 保持 False，由有效掩码另行排除）."""
    from .config import DEFAULT_VALUES

    m = pd.DataFrame(index=Y.index)
    for c in TARGETS:
        m[c] = Y[c].notna() & np.isclose(Y[c], DEFAULT_VALUES[c])
    return m


def transform_target(c: str, y: pd.Series) -> pd.Series:
    """PERM 在对数空间回归，其余保持原值."""
    if c == "PERM":
        return np.log10(y.clip(lower=EPS))
    return y


def inverse_transform_target(c: str, y: pd.Series) -> pd.Series:
    """把模型输出反变换回物理量."""
    if c == "PERM":
        return np.power(10.0, y).clip(lower=EPS)
    return y

