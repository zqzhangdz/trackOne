"""官方评价指标复现."""

from __future__ import annotations

import numpy as np

from .config import EPS, TOLERANCE, WEIGHTS, TARGETS


def _clip_acc(pred: np.ndarray, true: np.ndarray, tol: float) -> float:
    """普通线性空间的截断准确率."""
    denom = tol * (np.abs(true) + EPS)
    return float(np.mean(np.maximum(0.0, 1.0 - np.abs(pred - true) / denom)))


def _clip_acc_log10(pred: np.ndarray, true: np.ndarray, tol: float) -> float:
    """log10 空间的截断准确率（用于 PERM）."""
    p = np.log10(np.maximum(pred, EPS))
    t = np.log10(np.maximum(true, EPS))
    denom = tol * (np.abs(t) + EPS)
    return float(np.mean(np.maximum(0.0, 1.0 - np.abs(p - t) / denom)))


def acc_single(c: str, pred: np.ndarray, true: np.ndarray) -> float:
    """单个目标的截断准确率（跳过无效值）."""
    p = np.asarray(pred, dtype=float)
    t = np.asarray(true, dtype=float)
    m = ~(np.isnan(p) | np.isnan(t))
    if m.sum() == 0:
        return 0.0
    p, t = p[m], t[m]
    tol = TOLERANCE[c]
    return _clip_acc_log10(p, t, tol) if c == "PERM" else _clip_acc(p, t, tol)


def acc_by_target(pred: dict[str, np.ndarray], true: dict[str, np.ndarray]) -> dict[str, float]:
    """计算三个目标的准确率."""
    return {c: acc_single(c, pred[c], true[c]) for c in TARGETS}


def total_score(pred: dict[str, np.ndarray], true: dict[str, np.ndarray]) -> float:
    """官方百分制总分."""
    acc = acc_by_target(pred, true)
    return float(sum(WEIGHTS[c] * acc[c] for c in TARGETS) * 100.0)
