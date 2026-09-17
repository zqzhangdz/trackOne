"""两阶段模型：默认/真实分类 + 真实段回归；支持目标变换/样本加权/单调约束开关."""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

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


def _clf(params: dict | None = None) -> HistGradientBoostingClassifier:
    p = dict(DEFAULT_CLF_PARAMS)
    p.update(params or {})
    return HistGradientBoostingClassifier(loss="log_loss", random_state=42, **p)


def _reg(params: dict | None = None, monotonic: dict | None = None, columns=None) -> HistGradientBoostingRegressor:
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


def predict(model: dict, X: pd.DataFrame) -> pd.DataFrame:
    """基于加载的模型 dict 进行推理（兼容旧工件：缺 target_transform 时按 perm_log）."""
    mode = model.get("target_transform", "perm_log")
    out = pd.DataFrame(index=X.index)
    for c in TARGETS:
        p_def = model["classifiers"][c].predict_proba(X)[:, 1]
        is_def = p_def >= model["thresholds"][c]
        reg_raw = model["regressors"][c].predict(X)
        reg_val = inverse_transform_target(c, pd.Series(reg_raw, index=X.index), mode).to_numpy()
        lo, hi = OUTPUT_CLIP[c]
        out[c] = np.clip(np.where(is_def, DEFAULT_VALUES[c], reg_val), lo, hi)
    return out
