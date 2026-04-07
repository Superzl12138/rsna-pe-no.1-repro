if [[ "${SKIP_PROCESS_INPUT:-0}" != "1" ]]; then
  cd process_input
  sh run.sh
  cd ..
else
  echo "[INFO] 跳过 process_input，直接复用现有 pickle。"
fi
cd lung_localization/split2
sh run.sh
cd ..
cd ..
cd seresnext50
sh run.sh
cd ..
cd seresnext101
sh run.sh
cd ..
cd 2nd_level
sh run.sh
cd ..
