#!/usr/bin/env bash
# 远端 GPU 平台运行入口（手工启动 / 粘贴进平台「启动命令」，见 CLAUDE.md §12）。
# 用法: DATA_IN=<数据位置> bash tools/remote_run.sh <stage>
#   stage: tree | tcn1 | all | eval | package
# DATA_IN 三种形态（按出现顺序兜底，最后一个存在即用）:
#   1. tar.xz 文件路径（tools/remote_bundle.sh 的产物）
#   2. 含 train/ test/ 两个子目录的目录
#   3. 不设 → 用当前目录下的 ./train ./test
set -euo pipefail

STAGE="${1:-eval}"

# ---- 依赖安装（幂等；平台启动命令里也可以换自己的 pip 源）----
python -m pip install --quiet -r requirements.txt
python -c "import torch" 2>/dev/null || python -m pip install --quiet torch

# ---- 设备选择：有 CUDA 用 CUDA，否则 CPU ----
DEVICE="${DEVICE:-}"
if [ -z "$DEVICE" ]; then
  if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    DEVICE="cuda"
  else
    DEVICE="cpu"
  fi
fi
echo "[remote] device=$DEVICE"

# ---- 数据准备 ----
TRAIN_DIR=""
TEST_DIR=""
if [ -n "${DATA_IN:-}" ] && [ -e "$DATA_IN" ]; then
  if [ -f "$DATA_IN" ]; then
    mkdir -p data_x
    tar -xJf "$DATA_IN" -C data_x 2>/dev/null || tar -xf "$DATA_IN" -C data_x
    TRAIN_DIR=data_x/train
    TEST_DIR=data_x/test
  elif [ -d "$DATA_IN/train" ]; then
    TRAIN_DIR="$DATA_IN/train"
    TEST_DIR="$DATA_IN/test"
  fi
fi
if [ -z "$TRAIN_DIR" ] && [ -d ./train ]; then
  TRAIN_DIR=./train
  TEST_DIR=./test
fi
if [ -z "$TRAIN_DIR" ] || [ ! -d "$TRAIN_DIR" ]; then
  echo "未找到训练数据。请设置 DATA_IN=<tar.xz 或含 train/ 的目录>，或把 train/ test/ 放到当前目录" >&2
  exit 1
fi
echo "[remote] TRAIN_DIR=$TRAIN_DIR  TEST_DIR=${TEST_DIR:-<无>}"

case "$STAGE" in
  tree)   # 零 GPU：v10 树(weight_tol 三目标 + POR absolute_error+bump) + 组装(v9 融合配置)
    python train.py --data_dir "$TRAIN_DIR" --out_dir ./models --stage tree --weight_tol 1
    python train.py --stage assemble --per_target_w 0.50,0.50,0.75 \
      --thresholds 0.08,0.25,0.05 --with_pass2 0
    python experiments/apply_v9_config.py
    ;;
  tcn1)   # GPU：3 seed × 150 epoch（迭代 2 目标：修正 metric_loss 后重训 TCN，可选 +wellz）
    python train.py --data_dir "$TRAIN_DIR" --out_dir ./models --stage tcn1 \
      --epochs1 150 --seed_list 42,43,44 --device "$DEVICE" --tcn_wellz "${TCN_WELLZ:-1}"
    echo "tcn1 done -> models/.staging/tcn1.joblib（下载后本地 assemble）"
    ;;
  all)    # 全流程（tree -> tcn1 -> assemble）
    python train.py --data_dir "$TRAIN_DIR" --out_dir ./models --stage tree --weight_tol 1
    python train.py --data_dir "$TRAIN_DIR" --out_dir ./models --stage tcn1 \
      --epochs1 150 --seed_list 42,43,44 --device "$DEVICE" --tcn_wellz "${TCN_WELLZ:-1}"
    python train.py --stage assemble --per_target_w 0.50,0.50,0.75 \
      --thresholds 0.08,0.25,0.05 --with_pass2 0
    python experiments/apply_v9_config.py
    ;;
  eval)   # 预测 + 打包，产物在 workdir，用平台下载通道带回本地
    if [ ! -d "${TEST_DIR:-/nonexistent}" ]; then
      echo "eval 需要 TEST_DIR；请确认 DATA_IN 同时含 test/" >&2
      exit 1
    fi
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