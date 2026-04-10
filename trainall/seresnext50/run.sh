set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_DIR="${REPRO_OUTPUT_ROOT:-/data/zilizhu/PE/repro_outputs}/trainall/seresnext50"
mkdir -p "$LOG_DIR"
bash "$ROOT_DIR/prepare_output_links.sh" "$SCRIPT_DIR" "$LOG_DIR" weights features0

START_STEP="${TRAINALL_SERESNEXT50_START_STEP:-train}"

if [[ "$START_STEP" == "train" ]]; then
  python -m torch.distributed.launch --nproc_per_node=${NPROC_PER_NODE:-4} train0.py > "$LOG_DIR/train0.txt"
fi
python save_train_features0.py > "$LOG_DIR/save_train_features0.txt"
