set -euo pipefail

START_STAGE="${START_STAGE:-process_input}"

should_run_stage() {
  local stage="$1"
  local order=0
  local start_order=0
  case "$stage" in
    process_input) order=1 ;;
    lung_localization) order=2 ;;
    seresnext50) order=3 ;;
    seresnext101) order=4 ;;
    second_level) order=5 ;;
    *) echo "[ERROR] Unknown stage: $stage"; exit 1 ;;
  esac
  case "$START_STAGE" in
    process_input) start_order=1 ;;
    lung_localization) start_order=2 ;;
    seresnext50) start_order=3 ;;
    seresnext101) start_order=4 ;;
    second_level) start_order=5 ;;
    *) echo "[ERROR] Unknown START_STAGE: $START_STAGE"; exit 1 ;;
  esac
  [[ $order -ge $start_order ]]
}

if should_run_stage process_input; then
  cd process_input
  sh run.sh
  cd ..
fi
if should_run_stage lung_localization; then
  cd lung_localization/splitall
  sh run.sh
  cd ..
  cd ..
fi
if should_run_stage seresnext50; then
  cd seresnext50
  sh run.sh
  cd ..
fi
if should_run_stage seresnext101; then
  cd seresnext101
  sh run.sh
  cd ..
fi
if should_run_stage second_level; then
  cd 2nd_level
  sh run.sh
  cd ..
fi
