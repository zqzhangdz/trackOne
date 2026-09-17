"""主预测脚本：读取测试数据并生成 result.json.

用法:
    python predict.py --data_dir ./data --output ./result.json
"""

from __future__ import annotations

import argparse
import os

from src.feature import load_wells, make_features
from src.model import load_model, predict
from src.utils import load_yaml_config, write_result_json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="测试数据目录")
    ap.add_argument("--output", required=True, help="结果 JSON 输出路径")
    ap.add_argument("--model_dir", default=None, help="模型目录，默认取 config.yaml 或 ./models")
    ap.add_argument("--config", default="./config.yaml", help="配置文件路径")
    args = ap.parse_args()

    cfg = load_yaml_config(args.config)
    model_dir = args.model_dir or cfg.get("model_dir", "./models")
    model_file = cfg.get("model_file", "model.pkl")
    scaler_file = cfg.get("scaler_file", "scaler.pkl")
    model_name = cfg.get("modelName", "TwoStageHistGradientBoosting")
    version = cfg.get("version", "1.0")

    model_path = os.path.join(model_dir, model_file)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"未找到模型文件: {model_path}")
    model = load_model(model_path)

    expected_features = None
    scaler_path = os.path.join(model_dir, scaler_file)
    if os.path.exists(scaler_path):
        import joblib

        scaler = joblib.load(scaler_path)
        expected_features = scaler.get("feature_columns")

    feat_context = bool(cfg.get("feature_context", False))
    feat_priors = bool(cfg.get("feature_priors", False))

    wells = load_wells(args.data_dir, has_labels=False)
    result_data = []
    for df in wells:
        log_id = df["logId"].iloc[0]
        X = make_features(df, context=feat_context, priors=feat_priors)
        if expected_features is not None:
            X = X[expected_features]
        pred = predict(model, X)
        predictions = []
        for i in range(len(df)):
            predictions.append(
                {
                    "depth": round(float(df["DEPTH"].iloc[i]), 4),
                    "POR": round(float(pred["POR"].iloc[i]), 4),
                    "PERM": round(float(pred["PERM"].iloc[i]), 4),
                    "SW": round(float(pred["SW"].iloc[i]), 4),
                }
            )
        predictions.sort(key=lambda r: r["depth"])
        result_data.append({"logId": log_id, "predictions": predictions})

    result = {
        "modelId": "",
        "modelName": model_name,
        "version": version,
        "resultData": result_data,
    }
    write_result_json(args.output, result)
    n_points = sum(len(r["predictions"]) for r in result_data)
    print(f"已生成 {args.output}: {len(result_data)} 口井, {n_points} 个深度点")


if __name__ == "__main__":
    main()

