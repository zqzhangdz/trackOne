"""两阶段模型：默认/真实分类 + 真实段回归；支持目标变换/样本加权/单调约束开关."""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

# 注意: sklearn 采用懒加载（_clf/_reg 内部导入）。
# 原因: 本模块同时被 torch 训练脚本导入; sklearn 的 OpenMP 运行时与 torch
# 线程池同进程交互在 macOS 上会导致挂起/段错误（见 EXPERIMENTS.md）。
from .feature import inverse_transform_target, transform_target
from .utils import DEFAULT_VALUES, EPS, OUTPUT_CLIP, TARGETS

DEFAULT_CLF_PARAMS = dict(
    learning_rate=0.1, max_iter=300, max_leaf_nodes=31,
    min_samples_leaf=50, l2_regularization=1.0, class_weight="balanced",
)
DEFAULT_REG_PARAMS = dict(
    learning_rate=0.08, max_iter=400, max_leaf_nodes=31,
    min_samples_leaf=50, l2_regularization=1.0,
)
# POR 回归器的单调约束：DEN 负相关、AC 正相关（其余为 0）
POR_MONOTONIC = {"DEN": -1, "AC": 1}


def _clf(params: dict | None = None):
    from sklearn.ensemble import HistGradientBoostingClassifier

    p = dict(DEFAULT_CLF_PARAMS)
    p.update(params or {})
    return HistGradientBoostingClassifier(loss="log_loss", random_state=42, **p)


def _reg(params: dict | None = None, monotonic: dict | None = None, columns=None):
    from sklearn.ensemble import HistGradientBoostingRegressor

    p = dict(DEFAULT_REG_PARAMS)
    p.update(params or {})
    if monotonic and columns is not None:
        cst = [monotonic.get(c, 0) for c in columns]
        if any(cst):
            p["monotonic_cst"] = cst
    return HistGradientBoostingRegressor(loss="squared_error", random_state=42, **p)


class TwoStageModel:
    """逐目标：默认分类器 + 真实段回归器.

    选项:
      target_transform: "perm_log"（仅 PERM 取 log10）或 "all_log"（三目标全 log10）
      sample_weight: 真实段回归按 1/(|y|+eps) 加权（强调相对误差）
      monotonic: 对 POR 回归器施加 DEN/AC 单调约束
    """

    def __init__(
        self,
        thresholds: dict[str, float] | None = None,
        target_transform: str = "perm_log",
        sample_weight: bool = False,
        monotonic: bool = False,
        clf_params: dict | None = None,
        reg_params: dict | None = None,
    ):
        self.thresholds = thresholds or {c: 0.5 for c in TARGETS}
        self.target_transform = target_transform
        self.sample_weight = sample_weight
        self.monotonic = monotonic
        self.clf_params = clf_params
        self.reg_params = reg_params
        self.classifiers: dict[str, HistGradientBoostingClassifier] = {}
        self.regressors: dict[str, HistGradientBoostingRegressor] = {}

    def fit(self, X: pd.DataFrame, Y: pd.DataFrame, is_default: pd.DataFrame) -> "TwoStageModel":
        for c in TARGETS:
            valid = Y[c].notna().to_numpy()
            y = Y.loc[valid, c]
            def_ = is_default.loc[valid, c].astype(int)
            Xv = X.loc[valid]

            clf = _clf(self.clf_params)
            clf.fit(Xv, def_)
            self.classifiers[c] = clf

            real = def_ == 0
            mono = POR_MONOTONIC if (self.monotonic and c == "POR") else None
            reg = _reg(self.reg_params, mono, list(Xv.columns))
            yt = transform_target(c, y.loc[real], self.target_transform)
            if self.sample_weight:
                w = 1.0 / (y.loc[real].abs().to_numpy() + EPS)
                w = w / w.mean()
                reg.fit(Xv.loc[real], yt, sample_weight=w)
            else:
                reg.fit(Xv.loc[real], yt)
            self.regressors[c] = reg
        return self

    def to_dict(self) -> dict:
        return {
            "thresholds": self.thresholds,
            "classifiers": self.classifiers,
            "regressors": self.regressors,
            "target_transform": self.target_transform,
        }


def load_model(path: str) -> dict:
    return joblib.load(path)


def predict(model: dict, X: pd.DataFrame, X_tcn: pd.DataFrame | None = None) -> pd.DataFrame:
    """基于加载的模型 dict 进行推理.

    兼容旧工件（缺 target_transform 时按 perm_log）。若工件含 tcn_sd 且提供
    标准化后的 TCN 输入 X_tcn，则回归值在 log10 空间按 blend_w 融合树与 TCN。
    """
    mode = model.get("target_transform", "perm_log")
    out = pd.DataFrame(index=X.index)
    regs_log: list[dict[str, np.ndarray]] = []
    p_clf_tcn: dict[str, list[np.ndarray]] = {c: [] for c in TARGETS}
    dils = model.get("tcn_dilations")
    if X_tcn is not None:
        x0 = X_tcn.to_numpy(dtype=np.float32)
        # 多 seed 集成：tcn_sds 为状态字典列表（可含 pass2 通道）；单 seed 时退化为 tcn_sd
        sds = model.get("tcn_sds")
        if sds is None and model.get("tcn_sd") is not None:
            sds = [model["tcn_sd"]]
        if sds:
            o1 = None
            for sd in sds:
                o1 = numpy_forward_tcn(x0, sd, dils)
                regs_log.append({c: o1[c][:, 1] for c in TARGETS})
                for c in TARGETS:
                    p_clf_tcn[c].append(_sigmoid_np(o1[c][:, 0]))
            if model.get("tcn2_sd") is not None and o1 is not None:
                ch = np.stack([_causal_smooth9(o1[c][:, 1]) for c in TARGETS], axis=1)
                o2 = numpy_forward_tcn(
                    np.concatenate([x0, ch.astype(np.float32)], axis=1),
                    model["tcn2_sd"], model["tcn2_dilations"],
                )
                regs_log.append({c: o2[c][:, 1] for c in TARGETS})
    blend_w = model.get("blend_w")
    npaths = len(regs_log)
    if blend_w is None:
        blend_w = [1.0 - 0.5 * npaths] + [0.5 / max(1, npaths)] * npaths if npaths else [1.0]
    for c in TARGETS:
        p_def = model["classifiers"][c].predict_proba(X)[:, 1]
        # 可选：树 p_def 与 TCN p_def 概率平均（clf_tcn_w 为 TCN 权重，逐目标或标量）
        cw_all = model.get("clf_tcn_w")
        if cw_all is not None and p_clf_tcn[c]:
            cw = cw_all[c] if isinstance(cw_all, dict) else float(cw_all)
            if cw > 0:
                p_def = (1.0 - cw) * p_def + cw * np.mean(p_clf_tcn[c], axis=0)
        is_def = p_def >= model["thresholds"][c]
        reg_raw = model["regressors"][c].predict(X)
        reg_phys = inverse_transform_target(c, pd.Series(reg_raw, index=X.index), mode).to_numpy()
        reg_log = np.log10(np.maximum(reg_phys, EPS))
        if regs_log:
            bw = blend_w[c] if isinstance(blend_w, dict) else blend_w
            reg_log = bw[0] * reg_log + sum(w * r[c] for w, r in zip(bw[1:], regs_log))
            reg_phys = np.power(10.0, reg_log)
        lo, hi = OUTPUT_CLIP[c]
        out[c] = np.clip(np.where(is_def, DEFAULT_VALUES[c], reg_phys), lo, hi)
    return out


def _causal_smooth9(arr: np.ndarray) -> np.ndarray:
    """按井因果移动平均(win=9), 与 M4 pass2 训练通道一致."""
    csum = np.cumsum(np.concatenate([[0.0], arr.astype(np.float64)]))
    n = np.arange(1, arr.size + 1)
    w = np.minimum(n, 9)
    idx = np.maximum(n - 9, 0)
    return ((csum[1:] - csum[idx]) / w).astype(np.float32)


# ---------------------------------------------------------------------------
# 纯 numpy TCN 推理（提交包不依赖 torch；与 solution/tcn_model.py 的 torch 定义逐层一致）
# ---------------------------------------------------------------------------

def _sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _linear_np(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    return x @ w.T + b


def _gelu_np(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))


def _conv1d_np(x: np.ndarray, w: np.ndarray, b: np.ndarray, dilation: int) -> np.ndarray:
    """x: (T, C); 非因果 conv1d, kernel=5, padding=2*dilation, zero-pad."""
    T, C = x.shape
    pad = 2 * dilation
    xp = np.zeros((T + 2 * pad, C), dtype=x.dtype)
    xp[pad:pad + T] = x
    out = np.zeros((T, w.shape[0]), dtype=np.float32)
    for k in range(5):
        out += xp[k * dilation:k * dilation + T] @ w[:, :, k].T
    return out + b


def numpy_forward_tcn(x: np.ndarray, sd: dict, dilations: list[int]) -> dict[str, np.ndarray]:
    """x: (T, F) 标准化后特征; 返回 {c: (T, 2) [logit_default, reg_log10]}."""
    h = _linear_np(x, sd["in_proj.weight"], sd["in_proj.bias"])
    for i, d in enumerate(dilations):
        prefix = f"blocks.{i}."
        c = _conv1d_np(h, sd[prefix + "conv.weight"], sd[prefix + "conv.bias"], d)
        gw, gb = sd[prefix + "norm.weight"], sd[prefix + "norm.bias"]
        mu = c.mean(axis=1, keepdims=True)
        var = c.var(axis=1, keepdims=True)
        c = (c - mu) / np.sqrt(var + 1e-5) * gw + gb
        h = h + _gelu_np(c)
    gw, gb = sd["out_norm.weight"], sd["out_norm.bias"]
    mu = h.mean(axis=1, keepdims=True)
    var = h.var(axis=1, keepdims=True)
    h = _gelu_np((h - mu) / np.sqrt(var + 1e-5) * gw + gb)
    return {c: _linear_np(h, sd[f"heads.{c}.weight"], sd[f"heads.{c}.bias"]) for c in TARGETS}


# ---------------------------------------------------------------------------
# torch TCN 定义（懒加载：训练进程按需导入 torch，避免与 sklearn 的 OpenMP
# 运行时同进程冲突导致挂起/段错误，见 EXPERIMENTS.md）
# ---------------------------------------------------------------------------
_TCN_CACHE: dict = {}


def has_torch() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except Exception:
        return False


def get_tcn_classes() -> dict:
    """返回 {"TCNBlock", "TCNNet", "export_to_numpy"}；首次调用时导入 torch 并定义."""
    if _TCN_CACHE:
        return _TCN_CACHE
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class TCNBlock(nn.Module):
        def __init__(self, ch: int, dilation: int, dropout: float = 0.1):
            super().__init__()
            self.conv = nn.Conv1d(ch, ch, kernel_size=5, padding=2 * dilation, dilation=dilation)
            self.norm = nn.LayerNorm(ch)
            self.drop = nn.Dropout(dropout)

        def forward(self, x):
            h = self.conv(x).transpose(1, 2)
            h = self.drop(F.gelu(self.norm(h), approximate="tanh")).transpose(1, 2)
            return x + h

    class TCNNet(nn.Module):
        """非因果(双向) TCN：逐点输出 3 目标 [logit_default, reg_log10]."""

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

        def forward(self, x):
            h = self.in_proj(x).transpose(1, 2)
            for b in self.blocks:
                h = b(h)
            h = F.gelu(self.out_norm(h.transpose(1, 2)), approximate="tanh")
            return {c: self.heads[c](h) for c in TARGETS}

    def export_to_numpy(model) -> dict:
        return {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}

    _TCN_CACHE.update(TCNBlock=TCNBlock, TCNNet=TCNNet, export_to_numpy=export_to_numpy)
    return _TCN_CACHE
