"""两阶段模型：默认/真实分类 + 真实段回归."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from .config import DEFAULT_VALUES, OUTPUT_CLIP, TARGETS
from .data import inverse_transform_target, transform_target


def _clf() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.1,
        max_iter=300,
        max_leaf_nodes=31,
        min_samples_leaf=50,
        l2_regularization=1.0,
        class_weight="balanced",
        random_state=42,
    )


def _reg() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.08,
        max_iter=400,
        max_leaf_nodes=31,
        min_samples_leaf=50,
        l2_regularization=1.0,
        random_state=42,
    )


class TwoStageModel:
    """逐目标训练一个默认分类器 + 一个真实段回归器."""

    def __init__(self, thresholds: dict[str, float] | None = None):
        self.thresholds = thresholds or {c: 0.5 for c in TARGETS}
        self.classifiers: dict[str, HistGradientBoostingClassifier] = {}
        self.regressors: dict[str, HistGradientBoostingRegressor] = {}

    def fit(
        self,
        X: pd.DataFrame,
        Y: pd.DataFrame,
        is_default: pd.DataFrame,
    ) -> "TwoStageModel":
        for c in TARGETS:
            valid = Y[c].notna().to_numpy()
            y = Y.loc[valid, c]
            def_ = is_default.loc[valid, c].astype(int)
            Xv = X.loc[valid]

            clf = _clf()
            clf.fit(Xv, def_)
            self.classifiers[c] = clf

            real = def_ == 0
            if real.sum() == 0:
                raise RuntimeError(f"目标 {c} 没有任何真实段样本")
            reg = _reg()
            reg.fit(Xv.loc[real], transform_target(c, y.loc[real]))
            self.regressors[c] = reg
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=X.index)
        for c in TARGETS:
            p_def = self.classifiers[c].predict_proba(X)[:, 1]
            is_def = p_def >= self.thresholds[c]
            reg_raw = self.regressors[c].predict(X)
            reg_val = inverse_transform_target(c, pd.Series(reg_raw, index=X.index))

            lo, hi = OUTPUT_CLIP[c]
            pred = np.where(is_def, DEFAULT_VALUES[c], reg_val.to_numpy())
            out[c] = np.clip(pred, lo, hi)
        return out

