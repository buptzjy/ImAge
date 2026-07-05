#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/autodl-tmp/root}"
ENV_NAME="${ENV_NAME:-ImAge}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
CONDA_HOME="${CONDA_HOME:-${DATA_ROOT}/miniconda3}"
ENV_PREFIX="${ENV_PREFIX:-${CONDA_HOME}/envs/${ENV_NAME}}"
PROJECT_ROOT="${PROJECT_ROOT:-${DATA_ROOT}/ImAge}"
PIP_MIRROR="${PIP_MIRROR:-https://pypi.tuna.tsinghua.edu.cn/simple}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-${DATA_ROOT}/.cache/pip}"
UV_CACHE_DIR="${UV_CACHE_DIR:-${DATA_ROOT}/.cache/uv}"
TMPDIR="${TMPDIR:-${DATA_ROOT}/tmp}"

export CONDA_HOME
export PROJECT_ROOT
export PIP_MIRROR
export PIP_CACHE_DIR
export UV_CACHE_DIR
export TMPDIR

mkdir -p "${TMPDIR}" "${PIP_CACHE_DIR}" "${UV_CACHE_DIR}" "$(dirname "${CONDA_HOME}")" "${PROJECT_ROOT}"

log() {
  printf '\n[%s] %s\n' "$(date '+%F %T')" "$*"
}

have_cmd() {
  command -v "$1" >/dev/null 2>&1
}

ensure_conda() {
  if [[ -x "${CONDA_HOME}/bin/conda" ]]; then
    return
  fi

  log "conda not found, installing Miniconda to ${CONDA_HOME}"
  local installer="/tmp/miniconda.sh"

  if have_cmd wget; then
    wget -O "${installer}" https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  elif have_cmd curl; then
    curl -L https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o "${installer}"
  else
    echo "Neither wget nor curl is available; please install one of them first." >&2
    exit 1
  fi

  bash "${installer}" -b -p "${CONDA_HOME}"
}

init_conda_shell() {
  # shellcheck disable=SC1091
  source "${CONDA_HOME}/etc/profile.d/conda.sh"
}

configure_conda_channels() {
  log "configuring conda channels"
  conda config --remove-key channels >/dev/null 2>&1 || true
  conda config --add channels conda-forge
  conda config --set channel_priority flexible
}

accept_conda_tos_if_needed() {
  log "accepting conda Terms of Service if required"
  conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main >/dev/null 2>&1 || true
  conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r >/dev/null 2>&1 || true
}

detect_torch_channel() {
  local cuda_version major minor

  if ! have_cmd nvidia-smi; then
    echo "cpu"
    return
  fi

  cuda_version="$(nvidia-smi | sed -n 's/.*CUDA Version: \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -n 1)"
  if [[ -z "${cuda_version}" ]]; then
    echo "cpu"
    return
  fi

  major="${cuda_version%%.*}"
  minor="${cuda_version#*.}"

  if (( major > 12 )) || (( major == 12 && minor >= 8 )); then
    echo "cu128"
  elif (( major == 12 && minor >= 6 )); then
    echo "cu126"
  elif (( major == 12 && minor >= 4 )); then
    echo "cu124"
  elif (( major == 11 && minor >= 8 )); then
    echo "cu118"
  else
    echo "cpu"
  fi
}

create_env() {
  init_conda_shell
  configure_conda_channels
  accept_conda_tos_if_needed
  if [[ -d "${ENV_PREFIX}" ]]; then
    log "conda env already exists at ${ENV_PREFIX}, reusing it"
  else
    log "creating conda env at ${ENV_PREFIX} with Python ${PYTHON_VERSION}"
    conda create -y -p "${ENV_PREFIX}" "python=${PYTHON_VERSION}" pip
  fi
  conda activate "${ENV_PREFIX}"
  python -m pip install --upgrade pip setuptools wheel -i "${PIP_MIRROR}"
}

install_torch_stack() {
  local torch_channel
  torch_channel="$(detect_torch_channel)"

  if [[ "${torch_channel}" == "cpu" ]]; then
    log "installing CPU PyTorch stack"
    python -m pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 -i "${PIP_MIRROR}"
  else
    log "installing GPU PyTorch stack from channel ${torch_channel}"
    python -m pip install \
      torch==2.7.0 \
      torchvision==0.22.0 \
      torchaudio==2.7.0 \
      --index-url "https://download.pytorch.org/whl/${torch_channel}"
  fi
}

install_project_packages() {
  log "installing ImAge runtime packages"
  python -m pip install -i "${PIP_MIRROR}" \
    numpy==1.23.0 \
    scipy==1.13.1 \
    pandas==2.2.3 \
    scikit-learn==1.6.1 \
    pillow==11.0.0 \
    matplotlib==3.10.8 \
    tqdm==4.67.1 \
    einops==0.8.1 \
    pytorch-metric-learning==2.8.1 \
    transformers==4.57.3 \
    tokenizers==0.22.2 \
    huggingface-hub==0.36.0 \
    safetensors==0.7.0 \
    prettytable==3.17.0 \
    tensorboard==2.20.0 \
    gpustat==1.1.1
}

install_faiss() {
  log "trying to install faiss-gpu first"
  if python -m pip install -i "${PIP_MIRROR}" faiss-gpu==1.7.2; then
    return
  fi

  log "pip faiss-gpu failed, trying conda-forge faiss-gpu"
  if conda install -y -p "${ENV_PREFIX}" -c conda-forge faiss-gpu; then
    return
  fi

  log "faiss-gpu is unavailable on this machine, falling back to faiss-cpu"
  python -m pip install -i "${PIP_MIRROR}" faiss-cpu==1.8.0.post1
}

write_activation_hint() {
  local conda_init_line
  conda_init_line="source \"${CONDA_HOME}/etc/profile.d/conda.sh\" && conda activate \"${ENV_PREFIX}\""

  cat <<EOF

Install finished.

Activate the environment with:
${conda_init_line}

Recommended next steps:
1. Copy the ImAge project to: ${PROJECT_ROOT}
2. cd ${PROJECT_ROOT}
3. python -c "import torch, torchvision, faiss; print(torch.__version__, torch.version.cuda, torchvision.__version__)"
4. Current env path: ${ENV_PREFIX}
EOF
}

verify_install() {
  log "verifying imports"
  python - <<'PY'
import torch
import torchvision
import numpy
import pandas
import sklearn
import transformers
import faiss
print("python ok")
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
print("torchvision", torchvision.__version__)
print("numpy", numpy.__version__)
print("pandas", pandas.__version__)
print("sklearn", sklearn.__version__)
print("transformers", transformers.__version__)
print("faiss", getattr(faiss, "__version__", "unknown"))
PY
}

main() {
  ensure_conda
  init_conda_shell
  create_env
  install_torch_stack
  install_project_packages
  install_faiss
  verify_install
  write_activation_hint
}

main "$@"
