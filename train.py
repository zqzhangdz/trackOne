"""训练两阶段基线模型并保存到 models/ 目录.

用法:
    python train.py --data_dir ./train --out_dir ./models --cv_folds 5
"""

from __future__ import annotations

import argparse
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from src.feature import (
    default_mask,
    inverse_transform_target,
    load_wells,
    make_features,
    make_targets,
    transform_target,
)
from src.model import TwoStageModel, predict as model_predict
from src.utils import DEFAULT_VALUES, OUTPUT_CLIP, TARGETS, WEIGHTS, acc_single


THRESHOLD_GRID = np.arange(0.10, 0.91, 0.05)


def tune_thresholds(model: TwoStageModel, X_val: pd.DataFrame, Y_val: pd.DataFrame) -> dict[str, float]:
    best = {}
    for c in TARGETS:
        valid = Y_val[c].notna().to_numpy()
        if valid.sum() == 0:
            best[c] = 0.5
            continue
        p_def = model.classifiers[c].predict_proba(X_val)[:, 1]
        reg_val = inverse_transform_target(c, pd.Series(model.regressors[c].predict(X_val), index=X_val.index)).to_numpy()
        true = Y_val[c].to_numpy()[valid]
        p_def_v = p_def[valid]
        reg_v = reg_val[valid]
        best_th, best_acc = 0.5, -1.0
        for th in THRESHOLD_GRID:
            pred = np.clip(np.where(p_def_v >= th, DEFAULT_VALUES[c], reg_v), *OUTPUT_CLIP[c])
            a = acc_single(c, pred, true)
            if a > best_acc:
                best_acc, best_th = a, float(th)
        best[c] = best_th
    return best


def fit_and_score(X, Y, is_default, train_idx, val_idx):
    model = TwoStageModel().fit(X.iloc[train_idx], Y.iloc[train_idx], is_default.iloc[train_idx])
    th = tune_thresholds(model, X.iloc[val_idx], Y.iloc[val_idx])
    model.thresholds = th
    pred = model_predict(model.to_dict(), X.iloc[val_idx])
    acc = {c: acc_single(c, pred[c].to_numpy(), Y.iloc[val_idx][c].to_numpy()) for c in TARGETS}
    score = sum(WEIGHTS[c] * acc[c] for c in TARGETS) * 100.0
    return score, acc, th


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="./train")
    ap.add_argument("--out_dir", default="./models")
    ap.add_argument("--cv_folds", type=int, default=5)
    ap.add_argument("--random_state", type=int, default=42)
    args = ap.parse_args()

    wells = load_wells(args.data_dir, has_labels=True)
    df = pd.concat(wells, ignore_index=True)
    print(f"训练井数: {len(wells)}  总深度点: {len(df):,}")

    X = make_features(df)
    Y = make_targets(df)
    is_default = default_mask(Y)
    groups = df["logId"].to_numpy()

    gkf = GroupKFold(n_splits=args.cv_folds)
    fold_scores, fold_thresholds = [], []
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, Y, groups), start=1):
        score, acc, th = fit_and_score(X, Y, is_default, tr_idx, va_idx)
        fold_scores.append(score)
        fold_thresholds.append(th)
        print(
            f"Fold {fold}: total={score:.3f}  POR={acc['POR']:.4f}  "
            f"PERM={acc['PERM']:.4f}  SW={acc['SW']:.4f}  "
            f"th={ {k: round(v, 2) for k, v in th.items()} }"
        )

    mean_score = float(np.mean(fold_scores))
    print(f"平均 CV 总分: {mean_score:.3f}  (std={np.std(fold_scores):.3f})")

    final_th = {c: float(np.median([t[c] for t in fold_thresholds])) for c in TARGETS}
    print("最终阈值:", {k: round(v, 3) for k, v in final_th.items()})

    print("在全量训练集上训练最终模型 ...")
    final_model = TwoStageModel(thresholds=final_th).fit(X, Y, is_default)

    os.makedirs(args.out_dir, exist_ok=True)
    joblib.dump(final_model.to_dict(), os.path.join(args.out_dir, "model.pkl"))
    joblib.dump(
        {
            "feature_columns": list(X.columns),
            "default_values": DEFAULT_VALUES,
            "output_clip": OUTPUT_CLIP,
        },
        os.path.join(args.out_dir, "scaler.pkl"),
    )
    print(f"模型已保存到 {args.out_dir}")


if __name__ == "__main__":
    main()

