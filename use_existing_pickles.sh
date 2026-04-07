#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_DIR="$ROOT_DIR/trainval/process_input/split2"
SOURCE_DIR="${EXISTING_PICKLE_DIR:-/home/zilizhu/CAD_PE-main/process_input/split3}"

REQUIRED_FILES=(
  "series_dict.pickle"
  "image_dict.pickle"
  "series_list_train.pickle"
  "series_list_valid.pickle"
  "image_list_train.pickle"
  "image_list_valid.pickle"
)

if [[ ! -d "$SOURCE_DIR" ]]; then
  echo "[ERROR] 现有 pickle 目录不存在: $SOURCE_DIR"
  exit 1
fi

mkdir -p "$TARGET_DIR"

for f in "${REQUIRED_FILES[@]}"; do
  if [[ ! -e "$SOURCE_DIR/$f" ]]; then
    echo "[ERROR] 缺少 $SOURCE_DIR/$f"
    exit 1
  fi
  ln -sfn "$SOURCE_DIR/$f" "$TARGET_DIR/$f"
done

echo "已复用现有 pickle:"
echo "  来源: $SOURCE_DIR"
echo "  目标: $TARGET_DIR"
