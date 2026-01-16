#!/usr/bin/env bash
# 下载 CCPD2020（如失败请改用本地已有数据或手动下载）。
# 默认放在 $1/raw_ccpd，如果 $1 未指定则用当前目录/raw_ccpd。

set -euo pipefail

OUT_ROOT=${1:-"./raw_ccpd"}
mkdir -p "$OUT_ROOT"

echo "[INFO] 目标目录: $OUT_ROOT"
echo "[INFO] 尝试使用 gdown，如无法连接请手动下载 CCPD2020 并解压到该目录。"

# 常见公开链接（可能需要科学上网，若失效请自行替换）
# Google Drive (约 20GB):
#   gdown 1x8v18cO_tQkfpQOy3O0VnfyqCk41sJLj -O CCPD2020.tar.xz
# 或分包：
#   gdown 1LSRr_4E9OBWu95BJhP_YL-OdGTu_5fZd -O CCPD2020-part1.tar
#   gdown 12Z8nwK6JqqPiR-qsJNZS6gD4xyiABT3U -O CCPD2020-part2.tar
# Tsinghua 云（需登录）：https://cloud.tsinghua.edu.cn/d/e7a7c6061c0d4e0c8bff/

pushd "$OUT_ROOT" >/dev/null

if [ ! -f "CCPD2020.tar.xz" ]; then
  echo "[INFO] 未检测到 CCPD2020.tar.xz，尝试拉取单包（可能较慢且不稳定）..."
  gdown 1x8v18cO_tQkfpQOy3O0VnfyqCk41sJLj -O CCPD2020.tar.xz || true
fi

if [ -f "CCPD2020.tar.xz" ]; then
  echo "[INFO] 解压 CCPD2020.tar.xz ..."
  tar -xf CCPD2020.tar.xz
  echo "[INFO] 解压完成，如无 train/val/test，请检查文件内容。"
else
  echo "[WARN] 未能自动下载。请手动下载 CCPD2020 至 $OUT_ROOT 并解压。"
  echo "结构示例："
  echo "  raw_ccpd/train/*.jpg"
  echo "  raw_ccpd/val/*.jpg"
  echo "  raw_ccpd/test/*.jpg"
fi

popd >/dev/null

