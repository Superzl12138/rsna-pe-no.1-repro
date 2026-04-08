#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

SPLIT_DIR="$ROOT_DIR/trainval/process_input/split2"
REQUIRED_FILES=(
  "series_dict.pickle"
  "image_dict.pickle"
  "series_list_train.pickle"
  "series_list_valid.pickle"
  "image_list_train.pickle"
  "image_list_valid.pickle"
)
BBOX_REQUIRED_FILES=(
  "$ROOT_DIR/trainval/lung_localization/split2/bbox_dict_train.pickle"
  "$ROOT_DIR/trainval/lung_localization/split2/bbox_dict_valid.pickle"
)

# 数据软链接
bash "$ROOT_DIR/setup_data_links.sh"

# 如果用户要求跳过 process_input，则检查或建立 split2 所需文件
if [[ "${SKIP_PROCESS_INPUT:-0}" == "1" ]]; then
  mkdir -p "$SPLIT_DIR"
  # 若提供了外部 pickle 目录，则创建到 split2 的软链接
  if [[ -n "${EXISTING_PICKLE_DIR:-}" ]]; then
    bash "$ROOT_DIR/use_existing_pickles.sh"
  fi
  # 校验必须文件是否存在
  for f in "${REQUIRED_FILES[@]}"; do
    if [[ ! -e "$SPLIT_DIR/$f" ]]; then
      echo "[ERROR] 缺少 $SPLIT_DIR/$f。"
      echo "如已生成于其他目录，请设置："
      echo "  export EXISTING_PICKLE_DIR=/path/to/your/pickles"
      echo "并重试："
      echo "  SKIP_PROCESS_INPUT=1 bash run_trainval.sh"
      exit 1
    fi
  done
fi

python3 "$ROOT_DIR/prepare_trainval_subset.py"

if [[ "${SKIP_LUNG_LOCALIZATION:-0}" == "1" ]]; then
  for f in "${BBOX_REQUIRED_FILES[@]}"; do
    if [[ ! -e "$f" ]]; then
      echo "[ERROR] 缺少 $f。"
      echo "当前设置了 SKIP_LUNG_LOCALIZATION=1，但后续图像级模型需要已有 bbox_dict 产物。"
      exit 1
    fi
  done
fi

cd "$ROOT_DIR/trainval"
bash run.sh
