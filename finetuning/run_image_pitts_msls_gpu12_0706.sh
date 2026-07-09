#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/media/data/zhangjingyi/ImAge"
GPU_DEVICES="${1:-1,2}"
PYTHON_BIN="${PYTHON_BIN:-python}"

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

run_variant() {
  local tmp_group="$1"
  local save_dir="$2"

  echo "[$(date '+%F %T')] start ImAge tmp_group=${tmp_group}, save_dir=${save_dir}, GPU=${GPU_DEVICES}"
  (
    cd "${PROJECT_ROOT}"
    CUDA_VISIBLE_DEVICES="${GPU_DEVICES}" "${PYTHON_BIN}" "${COMMON_ARGS[@]}" \
      --tmp_group "${tmp_group}" \
      --save_dir "${save_dir}"
  )
  echo "[$(date '+%F %T')] done ImAge tmp_group=${tmp_group}"
}

run_variant pitts image_mixed_resume_sfxl_pitts_0706_gpu12
run_variant msls image_mixed_resume_sfxl_msls_0706_gpu12
