#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 <current_training_pid> [gpu_devices]"
  echo "Example: $0 1234567 2,3"
  exit 1
fi

CURRENT_PID="$1"
GPU_DEVICES="${2:-2,3}"

PROJECT_ROOT="/media/data/zhangjingyi/ImAge"
PYTHON_BIN="${PYTHON_BIN:-python}"
TIMESTAMP="$(date +%m%d_%H%M%S)"
VARIANT_ORDER=(
  "all:image_mixed_resume_sfxl_tmp"
  "pitts:image_mixed_resume_sfxl_pitts"
  "msls:image_mixed_resume_sfxl_msls"
)

COMMON_ARGS=(
  -u finetuning/train_mixed_resume.py
  --aggregator image
  --resume_author /media/data/zhangjingyi/ImAge/module/ImAge_GSV.pth
  --training_dataset /data_nvme/zhangjingyi/Gsv_reflect/mixgsv
  --training_subsets default tmp
  --eval_datasets_folder /media/data1/chenshunpeng1/datasets
  --eval_dataset_names Msls_740 pitts30k
  --final_eval_dataset_names sped amstertime tokyo nordland svox
  --freeze_te 8
  --num_learnable_aggregation_tokens 8
  --train_batch_size 120
  --train_image_size 322 322
  --resize 322 322
  --lr 5e-6
  --epochs_num 20
  --warmup_epochs 5
  --real_to_synthetic 8
  --patience 5
)

wait_for_current_run() {
  echo "[$(date '+%F %T')] waiting for current training pid ${CURRENT_PID} to finish..."
  while kill -0 "${CURRENT_PID}" 2>/dev/null; do
    sleep 60
  done
  echo "[$(date '+%F %T')] current training pid ${CURRENT_PID} has finished."
}

run_variant() {
  local tmp_group="$1"
  local save_dir="$2"
  local log_path="${PROJECT_ROOT}/log_${save_dir}_${TIMESTAMP}.txt"

  echo "[$(date '+%F %T')] starting variant tmp_group=${tmp_group}, save_dir=${save_dir}"
  (
    cd "${PROJECT_ROOT}"
    CUDA_VISIBLE_DEVICES="${GPU_DEVICES}" "${PYTHON_BIN}" "${COMMON_ARGS[@]}" \
      --tmp_group "${tmp_group}" \
      --save_dir "${save_dir}"
  ) 2>&1 | tee "${log_path}"
  local exit_code="${PIPESTATUS[0]}"
  if [[ "${exit_code}" -ne 0 ]]; then
    echo "[$(date '+%F %T')] variant ${save_dir} failed with exit code ${exit_code}. log: ${log_path}"
    exit "${exit_code}"
  fi
  echo "[$(date '+%F %T')] variant ${save_dir} completed. log: ${log_path}"
}

run_all_variants() {
  local variant_spec tmp_group save_dir

  for variant_spec in "${VARIANT_ORDER[@]}"; do
    tmp_group="${variant_spec%%:*}"
    save_dir="${variant_spec#*:}"
    run_variant "${tmp_group}" "${save_dir}"
  done
}

run_occupy_loop() {
  local save_dir_prefix="image_mixed_resume_sfxl_tmp_occupy"
  local run_index=1

  echo "[$(date '+%F %T')] all three experiments are done. entering occupy-only loop on tmp_group=all"
  while true; do
    local save_dir="${save_dir_prefix}_${run_index}"
    echo "[$(date '+%F %T')] starting occupy-only run ${run_index}: ${save_dir}"
    (
      cd "${PROJECT_ROOT}"
      CUDA_VISIBLE_DEVICES="${GPU_DEVICES}" "${PYTHON_BIN}" "${COMMON_ARGS[@]}" \
        --tmp_group all \
        --epochs_num 999999 \
        --occupy_only \
        --disable_file_logging \
        --save_dir "${save_dir}"
    ) >/dev/null 2>&1
    local exit_code="$?"
    echo "[$(date '+%F %T')] occupy-only run ${run_index} exited with code ${exit_code}, restarting in 10s"
    run_index=$((run_index + 1))
    sleep 10
  done
}

wait_for_current_run
run_all_variants
run_occupy_loop
