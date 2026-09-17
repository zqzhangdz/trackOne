"""双向 TCN 序列模型：torch 训练定义 + 纯 numpy 推理导出（提交包不依赖 torch）."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

TARGETS = ["POR", "PERM", "SW"]


class TCNBlock(nn.Module):
    def __init__(self, ch: int, dilation: int, dropout: float = 0.1):
        super().__init__()
        self.conv = nn.Conv1d(ch, ch, kernel_size=5, padding=2 * dilation, dilation=dilation)
        self.norm = nn.LayerNorm(ch)  # 逐深度点、跨通道归一化，与 numpy 导出一致
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x).transpose(1, 2)
        h = self.drop(F.gelu(self.norm(h), approximate="tanh")).transpose(1, 2)
        return x + h


class TCNNet(nn.Module):
    """非因果(双向) TCN：逐点输出 3 目标的默认 logit 与 log10 空间回归值."""

    def __init__(self, n_features: int, ch: int = 128, dilations=(1, 2, 4, 8, 16), repeats: int = 2, dropout: float = 0.1):
        super().__init__()
        self.in_proj = nn.Linear(n_features, ch)
        blocks = []
        for _ in range(repeats):
            for d in dilations:
                blocks.append(TCNBlock(ch, d, dropout))
        self.blocks = nn.ModuleList(blocks)
        self.out_norm = nn.LayerNorm(ch)
        self.heads = nn.ModuleDict({c: nn.Linear(ch, 2) for c in TARGETS})

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        # x: (B, T, F) -> (B, C, T)
        h = self.in_proj(x).transpose(1, 2)
        for b in self.blocks:
            h = b(h)
        h = F.gelu(self.out_norm(h.transpose(1, 2)), approximate="tanh")  # (B, T, C)
        return {c: self.heads[c](h) for c in TARGETS}  # (B, T, 2)


def export_to_numpy(model: TCNNet) -> dict:
    sd = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
    return sd


def _conv1d_np(x: np.ndarray, w: np.ndarray, b: np.ndarray, dilation: int) -> np.ndarray:
    """x: (T, C) 非因果 conv1d, kernel=5, padding=2*dilation, zero-pad. w: (Cout, Cin, 5)."""
    T, C = x.shape
    pad = 2 * dilation
    xp = np.zeros((T + 2 * pad, C), dtype=x.dtype)
    xp[pad:pad + T] = x
    out = np.zeros((T, w.shape[0]), dtype=np.float32)
    for k in range(5):
        start = pad - 2 * dilation + k * dilation
        out += xp[start:start + T] @ w[:, :, k].T
    return out + b


def _linear_np(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    return x @ w.T + b


def numpy_forward(x: np.ndarray, sd: dict, dilations: list[int]) -> dict[str, np.ndarray]:
    h = _linear_np(x, sd["in_proj.weight"], sd["in_proj.bias"])
    for i, d in enumerate(dilations):
        prefix = f"blocks.{i}."
        c = _conv1d_np(h, sd[prefix + "conv.weight"], sd[prefix + "conv.bias"], d)
        gw = sd[prefix + "norm.weight"]
        gb = sd[prefix + "norm.bias"]
        mu = c.mean(axis=1, keepdims=True)
        var = c.var(axis=1, keepdims=True)
        c = (c - mu) / np.sqrt(var + 1e-5) * gw + gb
        h = h + _gelu_np(c)
    mu = h.mean(axis=1, keepdims=True)
    var = h.var(axis=1, keepdims=True)
    h = (h - mu) / np.sqrt(var + 1e-5) * sd["out_norm.weight"] + sd["out_norm.bias"]
    h = _gelu_np(h)
    return {c: _linear_np(h, sd[f"heads.{c}.weight"], sd[f"heads.{c}.bias"]) for c in TARGETS}


def _gelu_np(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))
