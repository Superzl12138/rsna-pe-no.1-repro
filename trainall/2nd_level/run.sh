set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_DIR="${REPRO_OUTPUT_ROOT:-/data/zilizhu/PE/repro_outputs}/trainall/2nd_level"
mkdir -p "$LOG_DIR"
bash "$ROOT_DIR/prepare_output_links.sh" "$SCRIPT_DIR" "$LOG_DIR" weights predictions

START_STEP="${TRAINALL_SECOND_LEVEL_START_STEP:-seresnext50_192}"

if [[ "$START_STEP" == "seresnext50_192" ]]; then
  CUDA_VISIBLE_DEVICES="${SECOND_LEVEL_GPU:-0}" python seresnext50_192.py > "$LOG_DIR/seresnext50_192.txt"
fi
CUDA_VISIBLE_DEVICES="${SECOND_LEVEL_GPU:-0}" python seresnext101_192.py > "$LOG_DIR/seresnext101_192.txt"
