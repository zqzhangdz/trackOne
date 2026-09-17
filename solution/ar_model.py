"""自回归因果 GRU：输入上一深度点目标反馈，显式利用沿深度平滑."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .config import DEFAULT_VALUES, TARGETS
from .data import inverse_transform_target


class ARSeqNet(nn.Module):
    def __init__(self, n_features: int, hidden: int = 128, n_layers: int = 1):
        super().__init__()
        self.gru = nn.GRU(
            n_features + len(TARGETS),
            hidden,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=False,
        )
        self.clf_heads = nn.ModuleDict({c: nn.Linear(hidden, 1) for c in TARGETS})
        self.reg_heads = nn.ModuleDict({c: nn.Linear(hidden, 1) for c in TARGETS})

    def forward(self, x: torch.Tensor, prev: torch.Tensor):
        inp = torch.cat([x, prev], dim=-1)
        h, _ = self.gru(inp)
        clf = {c: self.clf_heads[c](h).squeeze(-1) for c in TARGETS}
        reg = {c: self.reg_heads[c](h).squeeze(-1) for c in TARGETS}
        return clf, reg


def ar_predict(
    model: ARSeqNet,
    X: np.ndarray,
    thresholds: dict[str, float],
    reg_mean: dict[str, float],
    reg_std: dict[str, float],
    default_std: dict[str, float],
    device: torch.device,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    """自回归逐点推理，返回 (物理量预测, p_default, 标准化回归输出)."""
    model.eval()
    T = X.shape[0]
    p_def = {c: np.zeros(T, dtype=np.float32) for c in TARGETS}
    reg_out = {c: np.zeros(T, dtype=np.float32) for c in TARGETS}
    signal = np.zeros((T, len(TARGETS)), dtype=np.float32)
    h = None
    with torch.no_grad():
        for t in range(T):
            x_t = torch.from_numpy(X[t : t + 1]).to(device).unsqueeze(1)  # (1,1,F)
            if t == 0:
                prev = torch.zeros(1, 1, len(TARGETS), device=device)
            else:
                prev = torch.from_numpy(signal[t - 1 : t]).to(device).unsqueeze(1)
            inp = torch.cat([x_t, prev], dim=-1)
            out, h = model.gru(inp, h)
            for ci, c in enumerate(TARGETS):
                p = float(torch.sigmoid(model.clf_heads[c](out)).item())
                r = float(model.reg_heads[c](out).item())
                p_def[c][t] = p
                reg_out[c][t] = r
                signal[t, ci] = default_std[c] if p >= thresholds[c] else r

    pred: dict[str, np.ndarray] = {}
    for c in TARGETS:
        r_phys = inverse_transform_target(
            c, pd.Series(reg_out[c] * reg_std[c] + reg_mean[c])
        ).to_numpy()
        pred[c] = np.where(p_def[c] >= thresholds[c], DEFAULT_VALUES[c], r_phys)
    return pred, p_def, reg_out


def apply_ar_thresholds(
    p_def: dict[str, np.ndarray],
    reg_out: dict[str, np.ndarray],
    thresholds: dict[str, float],
    reg_mean: dict[str, float],
    reg_std: dict[str, float],
) -> dict[str, np.ndarray]:
    """基于缓存的原始输出应用阈值，得到物理量预测."""
    pred = {}
    for c in TARGETS:
        r_phys = inverse_transform_target(
            c, pd.Series(reg_out[c] * reg_std[c] + reg_mean[c])
        ).to_numpy()
        pred[c] = np.where(p_def[c] >= thresholds[c], DEFAULT_VALUES[c], r_phys)
    return pred
