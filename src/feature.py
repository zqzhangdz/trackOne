"""数据读取、哨兵值处理与特征工程."""

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


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    X = df[FEATURES].copy()
    for c in FEATURES:
        X.loc[_is_sentinel(X[c]), c] = np.nan
    X["CNL"] = X["CNL"].clip(CNL_LOW, CNL_HIGH)
    X["logRT"] = np.log10(X["RT"].clip(lower=EPS))
    X["logRXO"] = np.log10(X["RXO"].clip(lower=EPS))
    X["RT_over_RXO"] = X["RT"] / X["RXO"].clip(lower=EPS)
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


def transform_target(c: str, y: pd.Series) -> pd.Series:
    if c == "PERM":
        return np.log10(y.clip(lower=EPS))
    return y


def inverse_transform_target(c: str, y: pd.Series) -> pd.Series:
    if c == "PERM":
        return np.power(10.0, y).clip(lower=EPS)
    return y

