#!/usr/bin/env bash
# 本地打包数据供手动上传到远端平台（数据在 .gitignore，不走 git）。
# 用法: bash tools/remote_bundle.sh [输出文件名]
set -euo pipefail
cd "$(dirname "$0")/.."

OUT="${1:-remote_data_$(date +%Y%m%d).tar.xz}"
if [ ! -d train ] || [ ! -d test ]; then
  echo "缺少 train/ 或 test/ 目录" >&2
  exit 1
fi
tar -cJf "$OUT" train test
echo "已生成 $OUT ($(du -h "$OUT" | cut -f1))"
echo "下一步: 在平台手动上传该 tar.xz，把落盘路径填进启动命令的 DATA_IN="