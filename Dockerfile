# Dockerfile — vLLM (NVFP4 stack) image
#
# All software inputs are pinned (known-good production set, 2026-08-15):
#   base image  nvidia/cuda:13.3.1-devel-ubuntu22.04
#   uv          0.12.5
#   python      3.13.15
#   packages    requirements.lock (full freeze — see its header for update notes)
FROM nvidia/cuda:13.3.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/root/.local/bin:$PATH"

# CUDA environment variables (nvcc + headers are needed to JIT the kernels)
ENV CUDA_HOME="/usr/local/cuda"
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

# Install curl for uv and clean up right away to keep the image small
RUN apt-get update && \
    apt-get install -y curl build-essential wget ca-certificates && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install pinned uv 0.12.5 (x86_64 release binary)
RUN curl -Lfsso /tmp/uv.tar.gz \
        https://github.com/astral-sh/uv/releases/download/0.12.5/uv-x86_64-unknown-linux-gnu.tar.gz && \
    tar -xzf /tmp/uv.tar.gz -C /tmp && \
    mv /tmp/uv-x86_64-unknown-linux-gnu/uv /root/.local/bin/uv && \
    rm -rf /tmp/uv.tar.gz /tmp/uv-x86_64-unknown-linux-gnu

WORKDIR /app

# Create the venv with pinned Python 3.13.15
RUN uv python install 3.13.15 && \
    uv venv /app/unsloth-nvfp4-env --python 3.13.15

# Use the venv for all subsequent commands
ENV VIRTUAL_ENV="/app/unsloth-nvfp4-env"
ENV PATH="/app/unsloth-nvfp4-env/bin:$PATH"

# Install the pinned package set (full lock), then clean the uv cache
# (saves disk space!). --torch-backend=auto resolves the +cu132 torch wheels.
COPY requirements.lock /tmp/requirements.lock
RUN uv pip install -r /tmp/requirements.lock --torch-backend=auto && \
    uv cache clean && \
    rm -f /tmp/requirements.lock

ENTRYPOINT ["vllm", "serve"]
