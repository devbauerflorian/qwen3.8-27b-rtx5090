# Qwen3.8-27B (NVFP4) on a single RTX 5090 with vLLM

Self-contained Docker deployment of
[**Qwen3.8-27B**](https://huggingface.co/Qwen/Qwen3.8-27B) in
[NVFP4 4-bit quantization](https://huggingface.co/gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090),
served by [vLLM](https://github.com/vllm-project/vllm) with an OpenAI-compatible
API — one 32 GB card, 256K-token context, auto tool choice included.

## Highlights

- Model: [Qwen3.8-27B-NVFP4-RTX5090](https://huggingface.co/gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090)
  — ModelOpt NVFP4 export tuned for a single RTX 5090
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
#    (public model, no HF_TOKEN needed)
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
| `--gpu-memory-utilization` | `0.96` | upstream card benchmarks at `0.97` (KV pool ≈ 276K tokens, full 256K); `0.90` holds only ~205K |
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
```

## Host requirements

The container is fully self-contained (it ships its own CUDA toolkit). On the
host you only need the NVIDIA driver and the NVIDIA container toolkit, which
`setup.sh` installs.

## Model

Weights: [gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090](https://huggingface.co/gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090)
— a **GeForce RTX 5090-specific** NVFP4 export of
[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), quantized with
[NVIDIA Model Optimizer](https://github.com/NVIDIA/Model-Optimizer)
(NVFP4 W4A4, group size 16, FP8 KV).

| | |
|---|---|
| Weights | ~20.6 GB on disk (3 shards), ~18.8 GB in VRAM |
| Quantization | ModelOpt NVFP4 W4A4 + FP8 KV cache |
| Context | full 262,144 tokens fit in 32 GB (FP8 KV pool ≈ 276K tokens) |
| Hardware | Blackwell tensor cores only — Hopper can load the files but cannot run NVFP4 |
| License | Apache-2.0 (same as the base model) |

Numbers from the [model card](https://huggingface.co/gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090):
~80 tok/s single-stream decode, up to ~1,030 tok/s aggregate at 16 concurrent
requests, 5/5 tool-call smoke tests, and accuracy on par with
[Unsloth's NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4) (75% on a
20×3-item smoke). The card also announces a follow-up weight drop with higher
accuracy/faster decode for the same 32 GB envelope — worth re-checking.

The weights are pulled into `./models/qwen3.8-27b-nvfp4/` by `setup.sh` and are
**not** part of this repository; this repo's tooling is MIT-licensed (see `LICENSE`).
