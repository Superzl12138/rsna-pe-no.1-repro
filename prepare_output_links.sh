#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <stage_dir> <output_dir> <name1> [name2 ...]"
  exit 1
fi

STAGE_DIR="$1"
OUTPUT_DIR="$2"
shift 2

mkdir -p "$OUTPUT_DIR"

for name in "$@"; do
  local_path="$STAGE_DIR/$name"
  target_path="$OUTPUT_DIR/$name"
  mkdir -p "$target_path"

  if [[ -L "$local_path" ]]; then
    ln -sfn "$target_path" "$local_path"
    continue
  fi

  if [[ -d "$local_path" ]]; then
    shopt -s dotglob nullglob
    files=("$local_path"/*)
    if (( ${#files[@]} > 0 )); then
      cp -a "$local_path"/. "$target_path"/
    fi
    shopt -u dotglob nullglob
    rm -rf "$local_path"
  elif [[ -e "$local_path" ]]; then
    echo "[ERROR] $local_path exists and is not a directory."
    exit 1
  fi

  ln -sfn "$target_path" "$local_path"
done
