"""⚠️ 已作废 —— 请勿用于任何评估或调参决策。

2026-09-26 核验官方公式原文后确认：本文件的 PERM 口径是错的。
本文件把 PERM 容差写成 0.08 * |log10 y|（实际约 0.02–0.36 decade），
而官方公式是 `1 - |log10(max(ŷ/y, ε))|`，**恒为 1.0 decade 的比值容差、无分母**。
旧口径比官方严 3～20 倍，会让 PERM 分数系统性偏低、Total 少算约 4.5 分。

**权威实现是 `src/utils.py` 的 `acc_single` / `total_score`（官方口径）。**
本文件只剩两个已否决的历史脚本 `train_ar.py` / `train_seq.py`（GRU / 自回归路线）在引用，
不参与当前交付，保留仅供追溯。
"""

"""官方评价指标复现（旧拷贝）."""

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
