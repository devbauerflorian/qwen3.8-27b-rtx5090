# Dockerfile
FROM nvidia/cuda:13.3.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/root/.local/bin:$PATH"

# CUDA Environment Variables
ENV CUDA_HOME="/usr/local/cuda"
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

# Install curl for uv and clean up right away to keep the image small
RUN apt-get update && \
    apt-get install -y curl build-essential wget ca-certificates && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

WORKDIR /app

# Create the venv with Python 3.13
RUN uv python install 3.13 && \
    uv venv /app/unsloth-nvfp4-env --python 3.13

# Use the venv for all subsequent commands
ENV VIRTUAL_ENV="/app/unsloth-nvfp4-env"
ENV PATH="/app/unsloth-nvfp4-env/bin:$PATH"

# Install the vLLM packages and clean the uv cache afterwards (saves disk space!)
RUN uv pip install "vllm>=0.25.0" "flashinfer-python>=0.6.13" "nvidia-cutlass-dsl>=4.5.2" --torch-backend=auto && \
    uv cache clean

ENTRYPOINT ["vllm", "serve"]