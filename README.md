# Qwen3.8-27B (NVFP4) on a single RTX 5090 with vLLM

Self-contained Docker deployment of **Qwen3.8-27B** in **NVFP4 4-bit quantization**,
served by [vLLM](https://github.com/vllm-project/vllm) with an OpenAI-compatible
API — one 32 GB card, 256K-token context, auto tool choice included.

## Highlights

- 27B model running **on a single RTX 5090** (32 GB) thanks to NVFP4 weights
- 256K context length (`--max-model-len 262144`) with FP8 KV cache
- OpenAI-compatible API (`/v1/chat/completions`, ...) with `qwen3` reasoning
  parsing and XML tool calling
- One `setup.sh` + one `docker compose up -d --build` to go

## Requirements

| | |
|---|---|
| GPU | 1x NVIDIA RTX 5090 (32 GB VRAM) |
| System RAM | 64 GB recommended (CUDA graph builds are RAM-hungry) |
| Host | Ubuntu 24.04/26.04, Docker, NVIDIA driver (datacenter branch) |
| Disk | ~20 GB for the model weights, ~10 GB for the Docker image |

## Quickstart

```bash
# 1. Install the NVIDIA container toolkit and download the model (~20 GB)
#    If the model repository requires access, export HF_TOKEN first.
sudo ./setup.sh

# 2. Configure the API key (optional)
cp .env.example .env
# edit .env and set VLLM_API_KEY (leave empty to disable auth — LAN use only!)

# 3. Build and start (first build takes a while)
docker compose up -d --build

# 4. Check it serves
curl http://localhost:8020/v1/models
```

The service listens on `http://localhost:8020` (container port 8000) and is
served under the model name `qwen3.8-27b`.

### Using the API

```bash
curl http://localhost:8020/v1/chat/completions \
  -H "Authorization: Bearer $VLLM_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.8-27b",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 256
  }'
```

Any OpenAI SDK works, e.g. with Python:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8020/v1", api_key=os.environ["VLLM_API_KEY"])
resp = client.chat.completions.create(
    model="qwen3.8-27b",
    messages=[{"role": "user", "content": "Hello!"}],
)
```

## Configuration

Most knobs live in `docker-compose.yml`:

| Setting | Value | Notes |
|---|---|---|
| Host port | `8020` | container port `8000` is fixed |
| `VLLM_API_KEY` | from `.env` | empty = no authentication |
| GPU | 1x, `CUDA_VISIBLE_DEVICES=0` | single RTX 5090 |
| `MAX_JOBS` / `NVCC_THREADS` | `4` | lower these on machines with less RAM |
| `--gpu-memory-utilization` | `0.96` | VRAM headroom for the KV cache |
| `--max-model-len` | `262144` | 256K context |
| `--kv-cache-dtype` | `fp8` | halves KV cache VRAM, longer contexts |
| `--trust-remote-code` | — | required by the NVFP4 (modelopt) config |
| `shm_size` | `16gb` | needed for multi-process workers |

## Project layout

```
Dockerfile                          vLLM + flashinfer + CUTLASS DSL image (NVFP4)
docker-compose.yml                  service definition (ports, volumes, serve flags)
setup.sh                            host prerequisites + model download (~20 GB)
.env.example                        secret template (copy to .env)
models/qwen3.8-27b-nvfp4/           model weights (empty in git, filled by setup.sh)
scripts/install-host-cuda-toolkit.sh optional: CUDA toolkit on the host itself
```

## Optional: CUDA toolkit on the host

The container is fully self-contained; you only need the NVIDIA driver and
the container toolkit on the host (installed by `setup.sh`). If you additionally
want `nvcc` outside the container (e.g. to build kernels locally):

```bash
sudo ./scripts/install-host-cuda-toolkit.sh
```

## Model & licensing

Model weights are downloaded from `gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090`
and are **not** part of this repository. Check the weights' own license terms
before commercial use. This repository's tooling is MIT-licensed (see `LICENSE`).
