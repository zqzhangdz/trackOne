"""双向 TCN：torch 定义与 numpy 导出均单一实现源于 src.model（提交包不依赖本文件）."""

from __future__ import annotations

from src.model import get_tcn_classes, numpy_forward_tcn as numpy_forward  # noqa: F401

_tc = get_tcn_classes()
TCNNet = _tc["TCNNet"]
TCNBlock = _tc["TCNBlock"]
export_to_numpy = _tc["export_to_numpy"]
