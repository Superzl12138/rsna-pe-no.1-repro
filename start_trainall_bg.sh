#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs}"
mkdir -p "$LOG_DIR"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_FILE:-$LOG_DIR/trainall_${TIMESTAMP}.log}"
PID_FILE="$LOG_DIR/trainall_latest.pid"

nohup bash "$ROOT_DIR/run_trainall.sh" >"$LOG_FILE" 2>&1 &
PID=$!
echo "$PID" > "$PID_FILE"

echo "trainall 已在后台启动"
echo "PID: $PID"
echo "日志: $LOG_FILE"
echo "PID 文件: $PID_FILE"
