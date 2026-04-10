set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_DIR="${REPRO_OUTPUT_ROOT:-/data/zilizhu/PE/repro_outputs}/trainval/seresnext50"
mkdir -p "$LOG_DIR"
bash "$ROOT_DIR/prepare_output_links.sh" "$SCRIPT_DIR" "$LOG_DIR" weights features0

START_STEP="${SERESNEXT50_START_STEP:-train}"

should_run() {
  local step="$1"
  local order=0
  local start_order=0
  case "$step" in
    train) order=1 ;;
    valid) order=2 ;;
    save_valid_features) order=3 ;;
    save_train_features) order=4 ;;
    *) echo "[ERROR] Unknown seresnext50 step: $step"; exit 1 ;;
  esac
  case "$START_STEP" in
    train) start_order=1 ;;
    valid) start_order=2 ;;
    save_valid_features) start_order=3 ;;
    save_train_features) start_order=4 ;;
    *) echo "[ERROR] Unknown SERESNEXT50_START_STEP: $START_STEP"; exit 1 ;;
  esac
  [[ $order -ge $start_order ]]
}

if should_run train; then
  python -m torch.distributed.launch --nproc_per_node=${NPROC_PER_NODE:-4} train0.py > "$LOG_DIR/train0.txt"
fi
if should_run valid; then
  python valid0.py > "$LOG_DIR/valid0.txt"
fi
if should_run save_valid_features; then
  python save_valid_features0.py > "$LOG_DIR/save_valid_features0.txt"
fi
if should_run save_train_features; then
  python save_train_features0.py > "$LOG_DIR/save_train_features0.txt"
fi
