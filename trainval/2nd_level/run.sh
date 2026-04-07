LOG_DIR="${REPRO_OUTPUT_ROOT:-/data/zilizhu/PE/repro_outputs}/trainval/2nd_level"
mkdir -p "$LOG_DIR"

CUDA_VISIBLE_DEVICES="${SECOND_LEVEL_GPU:-0}" python seresnext50_128.py > "$LOG_DIR/seresnext50_128.txt"
CUDA_VISIBLE_DEVICES="${SECOND_LEVEL_GPU:-0}" python seresnext101_128.py > "$LOG_DIR/seresnext101_128.txt"
CUDA_VISIBLE_DEVICES="${SECOND_LEVEL_GPU:-0}" python seresnext50_192.py > "$LOG_DIR/seresnext50_192.txt"
CUDA_VISIBLE_DEVICES="${SECOND_LEVEL_GPU:-0}" python seresnext101_192.py > "$LOG_DIR/seresnext101_192.txt"
python prediction_correction.py > "$LOG_DIR/prediction_correction.txt"
