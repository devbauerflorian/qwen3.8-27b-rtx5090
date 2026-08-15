#!/usr/bin/env bash
# setup.sh
set -euo pipefail

# Install nvidia docker container toolkit
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
sudo apt-get update && sudo apt-get install -y --no-install-recommends \
   ca-certificates \
   curl \
   gnupg2
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo sed -i -e '/experimental/ s/^#//g' /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
export NVIDIA_CONTAINER_TOOLKIT_VERSION=1.20.0-1
  sudo apt-get install -y \
      nvidia-container-toolkit=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
      nvidia-container-toolkit-base=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
      libnvidia-container-tools=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
      libnvidia-container1=${NVIDIA_CONTAINER_TOOLKIT_VERSION}
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Configure paths
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Pick up model selection from .env (same keys docker compose uses):
#   MODEL_REPO, MODEL_SUBDIR, SERVED_MODEL_NAME
if [ -f "${ROOT_DIR}/.env" ]; then
  set -a
  . "${ROOT_DIR}/.env"
  set +a
fi

MODEL_DIR="${MODEL_DIR:-${ROOT_DIR}/models}"
# Default model (Apache-2.0, RTX 5090 NVFP4 export of Qwen/Qwen3.8-27B):
#   https://huggingface.co/gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090
MODEL_REPO="${MODEL_REPO:-gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090}"
MODEL_SUBDIR="${MODEL_SUBDIR:-qwen3.8-27b-nvfp4}"
# Weights revision (known-good state, 2026-08-15) — pinned so a re-run always
# fetches exactly what was tested. Override via .env; set MODEL_REVISION=
# (empty) to always track the latest weights instead.
MODEL_REVISION="${MODEL_REVISION:-69274a0d8dff5dd35bcee8290612f71e03b6e981}"
VENV_DIR="${ROOT_DIR}/.venv_download"

mkdir -p "${MODEL_DIR}"

echo "Setup root: ${ROOT_DIR}"
echo "Model dir:  ${MODEL_DIR}/${MODEL_SUBDIR}"
echo "Setting up virtual environment..."

# Create the virtual environment if it doesn't exist
if [ ! -d "${VENV_DIR}" ]; then
  python3 -m venv "${VENV_DIR}"
fi

# Activate the virtual environment
source "${VENV_DIR}/bin/activate"

# Install huggingface_hub (pinned) if the new 'hf' CLI is not present
if ! command -v hf >/dev/null 2>&1; then
  echo "Installing huggingface_hub (pinned)..."
  pip install -U "huggingface_hub[cli]==1.27.0"
fi

echo "Starting download of ${MODEL_REPO}${MODEL_REVISION:+ @ ${MODEL_REVISION}} ..."

# Pinned, accelerated download via Xet (replacing hf_transfer)
REV_ARGS=()
[ -n "${MODEL_REVISION}" ] && REV_ARGS+=(--revision "${MODEL_REVISION}")
HF_XET_HIGH_PERFORMANCE=1 hf download \
  "${MODEL_REPO}" \
  ${REV_ARGS[@]+"${REV_ARGS[@]}"} \
  --local-dir "${MODEL_DIR}/${MODEL_SUBDIR}"

# Deactivate the virtual environment
deactivate

echo ""
echo "Download completed!"
echo "You can now start the model using 'docker compose up -d --build'."