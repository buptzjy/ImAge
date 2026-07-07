#!/usr/bin/env bash

set -euo pipefail

IMAGE_ROOT="/media/data/zhangjingyi/ImAge"
EDTFORMER_ROOT="/media/data/zhangjingyi/EDTformer"
SAGE_ROOT="/media/data/zhangjingyi/SAGE"
PYTHON="${PYTHON:-/media/data1/zhangjingyi/miniconda3/envs/ImAge/bin/python}"
DATASETS_ROOT="${DATASETS_ROOT:-/media/data1/chenshunpeng1/datasets}"
GPU="${GPU:-0}"
LOG_PATH="${LOG_PATH:-${IMAGE_ROOT}/log_eval_all_baselines_sage_lowmem.txt}"

BASELINES="${BASELINES:-${BASELINE:-image boq salad edtformer}}"
DATASET_GROUP="${DATASET_GROUP:-all}"
INFER_BATCH_SIZE="${INFER_BATCH_SIZE:-64}"
DATABASE_CHUNK_SIZE="${DATABASE_CHUNK_SIZE:-4096}"
MEMORY_MIN_AVAILABLE_KB="${MEMORY_MIN_AVAILABLE_KB:-$((30 * 1024 * 1024))}"
MEMORY_CHECK_INTERVAL_SECONDS="${MEMORY_CHECK_INTERVAL_SECONDS:-60}"
BOQ_PROJ_CHANNELS="${BOQ_PROJ_CHANNELS:-384}"
BOQ_OUTPUT_DIM="${BOQ_OUTPUT_DIM:-12288}"

IMAGE_RESUME="${IMAGE_RESUME:-/media/data/zhangjingyi/ImAge/module/ImAge_GSV.pth}"
BOQ_RESUME="${BOQ_RESUME:-/media/data/zhangjingyi/ImAge/module/dinov2_12288.pth}"
SALAD_RESUME="${SALAD_RESUME:-/media/data/zhangjingyi/ImAge/module/dino_salad.ckpt}"
EDTFORMER_RESUME="${EDTFORMER_RESUME:-/media/data/zhangjingyi/ImAge/module/EDTformer.pth}"

if [[ "${RUN_UNDER_NOHUP:-1}" == "1" && "${ALL_BASELINES_EVAL_NOHUP_LAUNCHED:-0}" != "1" ]]; then
  echo "[$(date '+%F %T')] launching all-baseline eval under nohup on GPU${GPU}..."
  echo "[$(date '+%F %T')] log: ${LOG_PATH}"
  ALL_BASELINES_EVAL_NOHUP_LAUNCHED=1 nohup bash "$0" "$@" > "${LOG_PATH}" 2>&1 &
  echo "[$(date '+%F %T')] background pid: $!"
  exit 0
fi

split_words() {
  local value="$1"
  value="${value//,/ }"
  printf '%s\n' ${value}
}

resolve_datasets() {
  if [[ -n "${DATASETS:-}" ]]; then
    split_words "${DATASETS}"
    return
  fi

  case "${DATASET_GROUP}" in
    core)
      printf '%s\n' Msls_740 pitts30k sped amstertime tokyo247 nordland
      ;;
    final)
      printf '%s\n' sped amstertime tokyo247 nordland
      ;;
    sfxl|sf_xl)
      printf '%s\n' SF_XL_v1 SF_XL_occlusion SF_XL_night
      ;;
    svox)
      printf '%s\n' SVOX-base SVOX-rain SVOX-sun SVOX-snow SVOX-night SVOX-overcast
      ;;
    sage)
      printf '%s\n' SF_XL_v1 SF_XL_occlusion SF_XL_night SVOX-base SVOX-rain SVOX-sun SVOX-snow SVOX-night SVOX-overcast
      ;;
    redbox)
      printf '%s\n' SF_XL_occlusion SF_XL_night SVOX-rain SVOX-sun SVOX-snow SVOX-night SVOX-overcast
      ;;
    all)
      printf '%s\n' Msls_740 pitts30k sped amstertime tokyo247 nordland SF_XL_v1 SF_XL_occlusion SF_XL_night SVOX-base SVOX-rain SVOX-sun SVOX-snow SVOX-night SVOX-overcast
      ;;
    *)
      echo "Unknown DATASET_GROUP=${DATASET_GROUP}. Use core/final/sfxl/svox/sage/redbox/all, or set DATASETS." >&2
      return 2
      ;;
  esac
}

memory_guard_wait() {
  local eval_pid="$1"
  local available_kb available_gib

  while kill -0 "${eval_pid}" 2>/dev/null; do
    available_kb="$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)"
    available_gib="$((available_kb / 1024 / 1024))"
    echo "[$(date '+%F %T')] memory guard: MemAvailable=${available_gib}GiB"
    free -h

    if [[ "${available_kb}" -lt "${MEMORY_MIN_AVAILABLE_KB}" ]]; then
      echo "[$(date '+%F %T')] MemAvailable below $((MEMORY_MIN_AVAILABLE_KB / 1024 / 1024))GiB; killing eval process group ${eval_pid}"
      kill -TERM -- "-${eval_pid}" 2>/dev/null || kill -TERM "${eval_pid}" 2>/dev/null || true
      sleep 10
      kill -KILL -- "-${eval_pid}" 2>/dev/null || kill -KILL "${eval_pid}" 2>/dev/null || true
      wait "${eval_pid}" 2>/dev/null || true
      return 137
    fi

    sleep "${MEMORY_CHECK_INTERVAL_SECONDS}"
  done

  wait "${eval_pid}"
}

run_eval_process() {
  local baseline="$1"
  local project_root="$2"
  local resume="$3"
  local tag="$4"
  shift 4
  local eval_args=("$@")
  local eval_pid
  local previous_dir

  echo
  echo "============================================================"
  echo "[$(date '+%F %T')] START baseline=${baseline} tag=${tag}"
  echo "project_root=${project_root}"
  echo "resume=${resume}"
  echo "eval_args=${eval_args[*]}"
  echo "gpu=${GPU}"
  echo "sage_root=${SAGE_ROOT}"
  echo "low_mem=SAGE-style efficient_ram_testing infer_batch_size=${INFER_BATCH_SIZE} database_chunk_size=${DATABASE_CHUNK_SIZE}"
  echo "memory_guard=kill if MemAvailable < $((MEMORY_MIN_AVAILABLE_KB / 1024 / 1024))GiB; check_interval=${MEMORY_CHECK_INTERVAL_SECONDS}s"
  echo "============================================================"

  previous_dir="$(pwd)"
  cd "${project_root}"
  CUDA_VISIBLE_DEVICES="${GPU}" PYTHONUNBUFFERED=1 setsid "${PYTHON}" -u eval.py "${eval_args[@]}" &
  eval_pid="$!"

  memory_guard_wait "${eval_pid}"
  cd "${previous_dir}"
  echo "[$(date '+%F %T')] END baseline=${baseline} tag=${tag}"
}

run_baseline() {
  local baseline="$1"
  shift
  local datasets=("$@")
  local tag="eval_${baseline}_${DATASET_GROUP}_sage_lowmem"

  case "${baseline}" in
    image|Image|ImAge)
      run_eval_process "image" "${IMAGE_ROOT}" "${IMAGE_RESUME}" "${tag}" \
        --eval_datasets_folder="${DATASETS_ROOT}" \
        --eval_dataset_name="${datasets[0]}" \
        --eval_dataset_names "${datasets[@]}" \
        --resize 322 322 \
        --backbone=dinov2 \
        --freeze_te=8 \
        --num_learnable_aggregation_tokens=8 \
        --num_register_tokens=4 \
        --infer_batch_size="${INFER_BATCH_SIZE}" \
        --num_workers=0 \
        --recall_values 1 5 10 \
        --efficient_ram_testing \
        --database_chunk_size="${DATABASE_CHUNK_SIZE}" \
        --resume="${RESUME:-${IMAGE_RESUME}}" \
        --save_dir="${tag}"
      ;;
    boq|BoQ)
      run_eval_process "boq" "${IMAGE_ROOT}" "${BOQ_RESUME}" "${tag}" \
        --eval_datasets_folder="${DATASETS_ROOT}" \
        --eval_dataset_name="${datasets[0]}" \
        --eval_dataset_names "${datasets[@]}" \
        --resize 322 322 \
        --backbone=dinov2 \
        --aggregator=boq \
        --freeze_te=10 \
        --num_learnable_aggregation_tokens=8 \
        --num_register_tokens=0 \
        --boq_proj_channels="${BOQ_PROJ_CHANNELS}" \
        --boq_output_dim="${BOQ_OUTPUT_DIM}" \
        --infer_batch_size="${INFER_BATCH_SIZE}" \
        --num_workers=0 \
        --recall_values 1 5 10 \
        --efficient_ram_testing \
        --database_chunk_size="${DATABASE_CHUNK_SIZE}" \
        --resume="${RESUME:-${BOQ_RESUME}}" \
        --save_dir="${tag}"
      ;;
    salad|SALAD)
      run_eval_process "salad" "${IMAGE_ROOT}" "${SALAD_RESUME}" "${tag}" \
        --eval_datasets_folder="${DATASETS_ROOT}" \
        --eval_dataset_name="${datasets[0]}" \
        --eval_dataset_names "${datasets[@]}" \
        --resize 322 322 \
        --backbone=dinov2 \
        --aggregator=salad \
        --freeze_te=10 \
        --num_learnable_aggregation_tokens=8 \
        --num_register_tokens=0 \
        --infer_batch_size="${INFER_BATCH_SIZE}" \
        --num_workers=0 \
        --recall_values 1 5 10 \
        --efficient_ram_testing \
        --database_chunk_size="${DATABASE_CHUNK_SIZE}" \
        --resume="${RESUME:-${SALAD_RESUME}}" \
        --save_dir="${tag}"
      ;;
    edtformer|EDTFormer|EDT)
      run_eval_process "edtformer" "${EDTFORMER_ROOT}" "${EDTFORMER_RESUME}" "${tag}" \
        --eval_datasets_folder="${DATASETS_ROOT}" \
        --eval_dataset_names "${datasets[@]}" \
        --resize 322 322 \
        --infer_batch_size="${INFER_BATCH_SIZE}" \
        --num_workers=0 \
        --recall_values 1 5 10 \
        --efficient_ram_testing \
        --database_chunk_size="${DATABASE_CHUNK_SIZE}" \
        --resume="${RESUME:-${EDTFORMER_RESUME}}" \
        --save_dir="${tag}"
      ;;
    *)
      echo "Unknown baseline '${baseline}'. Use image/boq/salad/edtformer." >&2
      return 2
      ;;
  esac
}

mapfile -t DATASET_LIST < <(resolve_datasets)
mapfile -t BASELINE_LIST < <(split_words "${BASELINES}")

echo "[$(date '+%F %T')] ALL BASELINE EVAL START"
echo "baselines=${BASELINE_LIST[*]}"
echo "dataset_group=${DATASET_GROUP}"
echo "datasets=${DATASET_LIST[*]}"
echo "gpu=${GPU}"
echo "log=${LOG_PATH}"
echo "python=${PYTHON}"
echo "datasets_root=${DATASETS_ROOT}"
echo "memory_guard_threshold=$((MEMORY_MIN_AVAILABLE_KB / 1024 / 1024))GiB"
echo "memory_check_interval=${MEMORY_CHECK_INTERVAL_SECONDS}s"
echo "SFXL/SVOX are forced through --efficient_ram_testing with chunked retrieval."

for baseline in "${BASELINE_LIST[@]}"; do
  run_baseline "${baseline}" "${DATASET_LIST[@]}"
done

echo
echo "[$(date '+%F %T')] ALL BASELINE EVAL DONE"
