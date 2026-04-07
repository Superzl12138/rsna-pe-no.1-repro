LOG_DIR="${REPRO_OUTPUT_ROOT:-/data/zilizhu/PE/repro_outputs}/trainval/seresnext50"
mkdir -p "$LOG_DIR"

python -m torch.distributed.launch --nproc_per_node=${NPROC_PER_NODE:-4} train0.py > "$LOG_DIR/train0.txt"
python valid0.py > "$LOG_DIR/valid0.txt"
python save_valid_features0.py > "$LOG_DIR/save_valid_features0.txt"
python save_train_features0.py > "$LOG_DIR/save_train_features0.txt"
