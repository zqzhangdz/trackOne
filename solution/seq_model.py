"""双向 GRU 序列模型：多任务（默认/真实分类 + 真实段回归）."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .config import DEFAULT_VALUES, TARGETS


class SeqNet(nn.Module):
    def __init__(self, n_features: int, hidden: int = 128, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.gru = nn.GRU(
            n_features,
            hidden,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        out_dim = hidden * 2
        self.clf_heads = nn.ModuleDict({c: nn.Linear(out_dim, 1) for c in TARGETS})
        self.reg_heads = nn.ModuleDict({c: nn.Linear(out_dim, 1) for c in TARGETS})

    def forward(self, x: torch.Tensor) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        h, _ = self.gru(x)  # (B, L, 2H)
        clf = {c: self.clf_heads[c](h).squeeze(-1) for c in TARGETS}  # (B, L)
        reg = {c: self.reg_heads[c](h).squeeze(-1) for c in TARGETS}
        return clf, reg


def smooth_l1(inputs: torch.Tensor, targets: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    return nn.functional.smooth_l1_loss(inputs, targets, beta=beta, reduction="mean")


def seq_loss(
    clf: dict[str, torch.Tensor],
    reg: dict[str, torch.Tensor],
    batch: dict,
    pos_weight: dict[str, float],
    betas: dict[str, float],
    w_clf: float = 2.0,
    w_reg: float = 1.0,
) -> torch.Tensor:
    total = 0.0
    for c in TARGETS:
        valid = batch[f"{c}_valid"]  # (B,L) bool
        isdef = batch[f"{c}_def"].float()
        reg_mask = batch[f"{c}_reg_mask"]  # (B,L) bool
        reg_target = batch[f"{c}_reg_target"]

        if valid.any():
            bce = nn.functional.binary_cross_entropy_with_logits(
                clf[c][valid], isdef[valid], pos_weight=torch.tensor(pos_weight[c], device=clf[c].device)
            )
        else:
            bce = 0.0

        if reg_mask.any():
            huber = smooth_l1(reg[c][reg_mask], reg_target[reg_mask], betas[c])
        else:
            huber = 0.0

        total = total + w_clf * bce + w_reg * huber
    return total


def _to_tensor(x: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.asarray(x))


class WellWindows:
    """按井采样定长窗口的轻量 Dataset."""

    def __init__(self, wells: list[dict], length: int = 512):
        self.wells = wells
        self.length = length
        self.lengths = np.array([len(w["X"]) for w in wells], dtype=np.int64)
        self.weights = self.lengths.astype(np.float64)
        self.weights /= self.weights.sum()

    def __len__(self) -> int:
        return len(self.wells) * 4

    def __getitem__(self, _: int):
        wi = int(np.random.choice(len(self.wells), p=self.weights))
        w = self.wells[wi]
        T = self.lengths[wi]
        if T <= self.length:
            start = 0
        else:
            start = int(np.random.randint(0, T - self.length))
        sl = slice(start, start + self.length)
        out = {"X": w["X"][sl]}
        for c in TARGETS:
            out[f"{c}_valid"] = w[f"{c}_valid"][sl]
            out[f"{c}_def"] = w[f"{c}_def"][sl]
            out[f"{c}_reg_mask"] = w[f"{c}_reg_mask"][sl]
            out[f"{c}_reg_target"] = w[f"{c}_reg_target"][sl]
        return out


def collate(batch: list[dict]) -> dict:
    out = {}
    for key in batch[0]:
        arr = np.stack([b[key] for b in batch])
        out[key] = torch.from_numpy(arr.astype(np.float32))
    # bool 掩码
    for c in TARGETS:
        for suffix in ["valid", "reg_mask"]:
            out[f"{c}_{suffix}"] = out[f"{c}_{suffix}"].bool()
    return out


def raw_predict_seq(
    model: SeqNet,
    X: np.ndarray,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """对整口井序列做推理，返回 p_default 与回归原始输出（未反变换）."""
    model.eval()
    p_def: dict[str, list[np.ndarray]] = {c: [] for c in TARGETS}
    reg: dict[str, list[np.ndarray]] = {c: [] for c in TARGETS}
    with torch.no_grad():
        x = torch.from_numpy(X.astype(np.float32)).unsqueeze(0).to(device)
        clf, reg_out = model(x)
        for c in TARGETS:
            p_def[c].append(torch.sigmoid(clf[c]).squeeze(0).cpu().numpy())
            reg[c].append(reg_out[c].squeeze(0).cpu().numpy())
    return (
        {c: np.concatenate(p_def[c]) for c in TARGETS},
        {c: np.concatenate(reg[c]) for c in TARGETS},
    )


def apply_thresholds(
    p_def: dict[str, np.ndarray],
    reg: dict[str, np.ndarray],
    thresholds: dict[str, float],
    reg_mean: dict[str, float] | None = None,
    reg_std: dict[str, float] | None = None,
) -> dict[str, np.ndarray]:
    """根据阈值合并默认常数与回归值，输出物理量."""
    from .data import inverse_transform_target
    import pandas as pd

    out = {}
    for c in TARGETS:
        r = reg[c]
        if reg_mean is not None and reg_std is not None:
            r = r * reg_std[c] + reg_mean[c]
        reg_v = inverse_transform_target(c, pd.Series(r)).to_numpy()
        out[c] = np.where(p_def[c] >= thresholds[c], DEFAULT_VALUES[c], reg_v)
    return out
