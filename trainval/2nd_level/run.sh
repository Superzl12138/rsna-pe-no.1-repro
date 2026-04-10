set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_DIR="${REPRO_OUTPUT_ROOT:-/data/zilizhu/PE/repro_outputs}/trainval/2nd_level"
mkdir -p "$LOG_DIR"
bash "$ROOT_DIR/prepare_output_links.sh" "$SCRIPT_DIR" "$LOG_DIR" weights predictions

START_STEP="${SECOND_LEVEL_START_STEP:-seresnext50_128}"

should_run() {
  local step="$1"
  local order=0
  local start_order=0
  case "$step" in
    seresnext50_128) order=1 ;;
    seresnext101_128) order=2 ;;
    seresnext50_192) order=3 ;;
    seresnext101_192) order=4 ;;
    prediction_correction) order=5 ;;
    *) echo "[ERROR] Unknown 2nd_level step: $step"; exit 1 ;;
  esac
  case "$START_STEP" in
    seresnext50_128) start_order=1 ;;
    seresnext101_128) start_order=2 ;;
    seresnext50_192) start_order=3 ;;
    seresnext101_192) start_order=4 ;;
    prediction_correction) start_order=5 ;;
    *) echo "[ERROR] Unknown SECOND_LEVEL_START_STEP: $START_STEP"; exit 1 ;;
  esac
  [[ $order -ge $start_order ]]
}

if should_run seresnext50_128; then
  CUDA_VISIBLE_DEVICES="${SECOND_LEVEL_GPU:-0}" python seresnext50_128.py > "$LOG_DIR/seresnext50_128.txt"
fi
if should_run seresnext101_128; then
  CUDA_VISIBLE_DEVICES="${SECOND_LEVEL_GPU:-0}" python seresnext101_128.py > "$LOG_DIR/seresnext101_128.txt"
fi
if should_run seresnext50_192; then
  CUDA_VISIBLE_DEVICES="${SECOND_LEVEL_GPU:-0}" python seresnext50_192.py > "$LOG_DIR/seresnext50_192.txt"
fi
if should_run seresnext101_192; then
  CUDA_VISIBLE_DEVICES="${SECOND_LEVEL_GPU:-0}" python seresnext101_192.py > "$LOG_DIR/seresnext101_192.txt"
fi
if should_run prediction_correction; then
  python prediction_correction.py > "$LOG_DIR/prediction_correction.txt"
fi
