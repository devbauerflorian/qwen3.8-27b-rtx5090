#!/usr/bin/env bash
# install-host-cuda-toolkit.sh
# Optional: install the CUDA Toolkit 13.3.1 on the host (Ubuntu 26.04, x86_64).
#
# NOTE: This is NOT required for the Docker workflow — the image ships its own
# CUDA toolkit and only the NVIDIA driver + container toolkit (installed by
# setup.sh) are needed on the host. Run this only if you want nvcc / CUDA
# headers available outside the container, e.g. to build kernels locally.
set -euo pipefail

CUDA_REPO_DEB="cuda-repo-ubuntu2604-13-3-local_13.3.1-610.43.02-1_amd64.deb"
PIN_URL="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2604/x86_64/cuda-ubuntu2604.pin"
DEB_URL="https://developer.download.nvidia.com/compute/cuda/13.3.1/local_installers/${CUDA_REPO_DEB}"

echo "Downloading CUDA repository pin and ${CUDA_REPO_DEB} ..."
wget -O cuda-ubuntu2604.pin "${PIN_URL}"
wget -O "${CUDA_REPO_DEB}" "${DEB_URL}"

sudo mv cuda-ubuntu2604.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo dpkg -i "${CUDA_REPO_DEB}"
sudo cp /var/cuda-repo-ubuntu2604-13-3-local/cuda-*-keyring.gpg /usr/share/keyrings/

sudo apt-get update
sudo apt-get -y install cuda-toolkit-13-3

echo "Done. Verify with: nvcc --version"
