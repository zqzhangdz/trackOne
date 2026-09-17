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


from src.model import numpy_forward_tcn as numpy_forward  # noqa: E402  (单一实现源)
