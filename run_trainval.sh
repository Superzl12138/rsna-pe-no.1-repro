#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "$ROOT_DIR/setup_data_links.sh"

cd "$ROOT_DIR/trainval"
bash run.sh
