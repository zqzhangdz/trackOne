"""双向 TCN：torch 定义与 numpy 导出均单一实现源于 src.model（提交包不依赖本文件）.

torch 类通过模块级 __getattr__ 懒加载：仅 import 本模块不会加载 torch，
避免与 sklearn 的 OpenMP 运行时同进程冲突（见 EXPERIMENTS.md 工程事故记录）。
"""

from __future__ import annotations

from src.model import get_tcn_classes
from src.model import numpy_forward_tcn as numpy_forward  # noqa: F401

_LAZY = ("TCNNet", "TCNBlock", "export_to_numpy")


def __getattr__(name: str):
    if name in _LAZY:
        return get_tcn_classes()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
