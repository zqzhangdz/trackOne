"""配置常量、评价指标与通用工具."""

from __future__ import annotations

import json
import os

import numpy as np

# 原始曲线列（测试集没有最后三列）
RAW_COLS = [
    "DEPTH", "GR", "PE", "SP", "CAL", "AC", "DEN", "CNL",
    "RXO", "RT", "DEVI", "AZIM", "BIT", "CASE",
]
TRAIN_COLS = RAW_COLS + ["POR", "PERM", "SW"]

# 实际参与建模的特征（CASE 缺失率 ~99%，剔除）
FEATURES = [
    "DEPTH", "GR", "PE", "SP", "CAL", "AC", "DEN", "CNL",
    "RXO", "RT", "DEVI", "AZIM", "BIT",
]
TARGETS = ["POR", "PERM", "SW"]

# 各目标在“非储层/默认段”的主导常数
DEFAULT_VALUES = {"POR": 0.1, "PERM": 0.01, "SW": 99.9}

# 数据中的缺失/异常哨兵值
SENTINEL_VALUES = [-99999.0, -9999.0, -2.534573822542048e37]

# 官方指标极小值
EPS = 1e-5

# CNL 物理截断（单位 %）
CNL_LOW, CNL_HIGH = 0.0, 100.0

# 输出 clip 范围
OUTPUT_CLIP = {
    "POR": (0.0, 50.0),
    "PERM": (EPS, 5000.0),
    "SW": (0.0, 100.0),
}

# 官方总分权重
WEIGHTS = {"POR": 0.3, "PERM": 0.35, "SW": 0.35}

# 各目标相对误差容差
TOLERANCE = {"POR": 0.08, "PERM": 0.08, "SW": 0.05}


def _clip_acc(pred: np.ndarray, true: np.ndarray, tol: float) -> float:
    denom = tol * (np.abs(true) + EPS)
    return float(np.mean(np.maximum(0.0, 1.0 - np.abs(pred - true) / denom)))


def _clip_acc_perm_official(pred: np.ndarray, true: np.ndarray) -> float:
    """官方 PERM 口径：mean(max(0, 1 - |log10(max(ŷ/y, ε))|))。

    官方公式**没有容差系数、没有分母**，等效容差恒为 1.0 decade。

    历史错误（2026-09-26 修正）：本函数旧实现按 0.08 * |log10 y| 作分母，
    实际容差只有约 0.02–0.36 decade —— 比官方严 3～20 倍。后果是 PERM 分数被
    系统性压低、Total 少算约 4.5 分（v6 旧口径 68.778 / 官方口径 73.310），
    并据此得出过"75 分线按实测不可达、不要再训练"的错误结论。
    旧口径保留在 `acc_single_legacy`，仅供追溯历史实验数字。
    """
    ratio = np.maximum(pred / np.maximum(true, EPS), EPS)
    return float(np.mean(np.maximum(0.0, 1.0 - np.abs(np.log10(ratio)))))


def _clip_acc_perm_legacy(pred: np.ndarray, true: np.ndarray, tol: float) -> float:
    """已作废的 PERM 旧口径，仅作历史数字复现用，勿用于新评估."""
    p = np.log10(np.maximum(pred, EPS))
    t = np.log10(np.maximum(true, EPS))
    denom = tol * (np.abs(t) + EPS)
    return float(np.mean(np.maximum(0.0, 1.0 - np.abs(p - t) / denom)))


def acc_single(c: str, pred: np.ndarray, true: np.ndarray) -> float:
    """单个目标的截断准确率（官方口径）."""
    p = np.asarray(pred, dtype=float)
    t = np.asarray(true, dtype=float)
    m = ~(np.isnan(p) | np.isnan(t))
    if m.sum() == 0:
        return 0.0
    p, t = p[m], t[m]
    if c == "PERM":
        return _clip_acc_perm_official(p, t)
    return _clip_acc(p, t, TOLERANCE[c])


def acc_single_legacy(c: str, pred: np.ndarray, true: np.ndarray) -> float:
    """2026-09-26 之前的口径（PERM 用 0.08 * |log10 y| 分母）。仅供追溯."""
    p = np.asarray(pred, dtype=float)
    t = np.asarray(true, dtype=float)
    m = ~(np.isnan(p) | np.isnan(t))
    if m.sum() == 0:
        return 0.0
    p, t = p[m], t[m]
    return _clip_acc_perm_legacy(p, t, TOLERANCE["PERM"]) if c == "PERM" \
        else _clip_acc(p, t, TOLERANCE[c])


def total_score(pred: dict[str, np.ndarray], true: dict[str, np.ndarray]) -> float:
    return float(sum(WEIGHTS[c] * acc_single(c, pred[c], true[c]) for c in TARGETS) * 100.0)


def load_yaml_config(path: str) -> dict:
    import yaml

    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def write_result_json(path: str, result: dict) -> None:
    out_dir = os.path.dirname(os.path.abspath(path))
    os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

