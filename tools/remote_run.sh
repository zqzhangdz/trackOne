#!/usr/bin/env bash
# 远端 GPU 平台运行入口（手工启动 / 粘贴进平台「启动命令」，见 CLAUDE.md §12）。
# 用法: DATA_IN=/code/workspace/remote_data.tar.xz bash tools/remote_run.sh <stage>
#   stage: tree | tcn1 | all | eval | package
# 本地开发时 DATA_IN 指向本地 tar.xz 或含 train/ test/ 的目录。
set -euo pipefail

STAGE="${1:-eval}"

# ---- 依赖安装（幂等，平台启动命令里也可以换自己的 pip 源）----
python -m pip install --quiet -r requirements.txt
python -c "import torch" 2>/dev/null || \
  python -m pip install --quiet torch --index-url https://download.pytorch.org/whl/cpu

# ---- 数据准备：优先用云盘挂载 -> 当前目录已有 train/test -> 本地 tar.xz ----
if [ -n "${DATA_IN:-}" ] && [ -f "$DATA_IN" ]; then
  mkdir -p data_x
  tar -xJf "$DATA_IN" -C data_x 2>/dev/null || tar -xf "$DATA_IN" -C data_x
  TRAIN_DIR=data_x/train
  TEST_DIR=data_x/test
elif [ -d /code/workspace/data_x/train ]; then
  TRAIN_DIR=/code/workspace/data_x/train
  TEST_DIR=/code/workspace/data_x/test
elif [ -d ./train ]; then
  TRAIN_DIR=./train
  TEST_DIR=./test
else
  echo "未找到数据：请设置 DATA_IN=<tar.xz 路径> 或把 train/ test/ 放到当前目录" >&2
  exit 1
fi
[ -d "$TRAIN_DIR" ] || { echo "缺少训练数据 $TRAIN_DIR" >&2; exit 1; }

case "$STAGE" in
  tree)   # 零 GPU：v10 树(squared_error pass-1 + POR absolute_error+bump weight_tol) + 组装(v9 融合配置)
    python train.py --data_dir "$TRAIN_DIR" --out_dir ./models --stage tree --weight_tol 1
    python train.py --stage assemble --per_target_w 0.50,0.50,0.75 \
      --thresholds 0.08,0.25,0.05 --with_pass2 0
    python experiments/apply_v9_config.py
    ;;
  tcn1)   # GPU：3 seed × 150 epoch（可选 --device cuda，默认让 torch 自选）
    python train.py --data_dir "$TRAIN_DIR" --out_dir ./models --stage tcn1 \
      --epochs1 150 --seed_list 42,43,44 ${TCN_DEVICE:---device cpu}
    echo "tcn1 done -> models/.staging/tcn1.joblib（下载后再本地 assemble / 直接 commit）"
    ;;
  all)    # 全流程（tree -> tcn1 -> assemble）
    python train.py --data_dir "$TRAIN_DIR" --out_dir ./models --stage tree --weight_tol 1
    python train.py --data_dir "$TRAIN_DIR" --out_dir ./models --stage tcn1 \
      --epochs1 150 --seed_list 42,43,44 ${TCN_DEVICE:---device cpu}
    python train.py --stage assemble --per_target_w 0.50,0.50,0.75 \
      --thresholds 0.08,0.25,0.05 --with_pass2 0
    python experiments/apply_v9_config.py
    ;;
  eval)   # 预测 + 打包，产出 zip；乘积在 workdir，用平台下载/云盘带回来
    python predict.py --data_dir "$TEST_DIR" --output ./result.json
    python tools/package_submission.py --workdir . --skip_predict 1 || true
    echo "产物: result.json / result.zip / submission_code.zip"
    ;;
  package)
    python tools/package_submission.py --workdir . --skip_predict 1 || true
    ;;
  *)
    echo "未知 stage: $STAGE（可用 tree|tcn1|all|eval|package）" >&2
    exit 1
    ;;
esac
echo "remote_run $STAGE done"