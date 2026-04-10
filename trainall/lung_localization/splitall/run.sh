set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
LOG_DIR="${REPRO_OUTPUT_ROOT:-/data/zilizhu/PE/repro_outputs}/trainall/lung_localization_splitall"
mkdir -p "$LOG_DIR"
bash "$ROOT_DIR/prepare_output_links.sh" "$SCRIPT_DIR" "$LOG_DIR" weights

python -m torch.distributed.launch --nproc_per_node=${NPROC_PER_NODE:-4} train0.py > "$LOG_DIR/train0.txt"
python -m torch.distributed.launch --nproc_per_node=${NPROC_PER_NODE:-4} train1.py > "$LOG_DIR/train1.txt"
python -m torch.distributed.launch --nproc_per_node=${NPROC_PER_NODE:-4} train2.py > "$LOG_DIR/train2.txt"
python save_bbox_train.py > "$LOG_DIR/save_bbox_train.txt"
