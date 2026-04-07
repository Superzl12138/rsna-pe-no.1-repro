#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INPUT_DIR="$ROOT_DIR/input"

PE_DATA_ROOT="${PE_DATA_ROOT:-/data/zilizhu/PE}"
PE_TRAIN_CSV="${PE_TRAIN_CSV:-$PE_DATA_ROOT/train.csv}"
PE_TRAIN_DIR="${PE_TRAIN_DIR:-$PE_DATA_ROOT/train}"

mkdir -p "$INPUT_DIR"

if [[ ! -f "$PE_TRAIN_CSV" ]]; then
  echo "缺少 train.csv: $PE_TRAIN_CSV"
  echo "请设置 PE_TRAIN_CSV 或 PE_DATA_ROOT"
  exit 1
fi

if [[ ! -d "$PE_TRAIN_DIR" ]]; then
  echo "缺少 DICOM 目录: $PE_TRAIN_DIR"
  echo "请设置 PE_TRAIN_DIR 或 PE_DATA_ROOT"
  exit 1
fi

ln -sfn "$PE_TRAIN_CSV" "$INPUT_DIR/train.csv"
ln -sfn "$PE_TRAIN_DIR" "$INPUT_DIR/train"

echo "已就绪:"
echo "  train.csv -> $PE_TRAIN_CSV"
echo "  train/    -> $PE_TRAIN_DIR"
