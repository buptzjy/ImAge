#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 <current_training_pid> [gpu_devices]"
  echo "Example: $0 1234567 2,3"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_SCRIPT="${SCRIPT_DIR}/run_image_sfxl_tmp_ablation.sh"

echo "[$(date '+%F %T')] ${0##*/} is now a compatibility wrapper."
echo "[$(date '+%F %T')] forwarding to ${TARGET_SCRIPT##*/} to run all variants and occupy loop."
exec bash "${TARGET_SCRIPT}" "$@"
