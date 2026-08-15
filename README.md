# Qwen3.8-27B (NVFP4) on a single RTX 5090 with vLLM

Self-contained Docker deployment of
[**Qwen3.8-27B**](https://huggingface.co/Qwen/Qwen3.8-27B) in
[NVFP4 4-bit quantization](https://huggingface.co/gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090),
served by [vLLM](https://github.com/vllm-project/vllm) with an OpenAI-compatible
API — one 32 GB card, 256K-token context, auto tool choice included.

## Purpose

Who is this for, and why? This repo makes one specific thing work, end to
end, on a **single consumer GPU (RTX 5090, 32 GB)**:

- a **27B model** (Qwen3.8-27B) at **NVFP4 4-bit** — ~18.8 GB in VRAM
- the **native 256K-token context** (FP8 KV cache) — most 4-bit recipes cap
  out far earlier on 32 GB
- an **OpenAI-compatible API** with reasoning + tool calling that any
  OpenAI-SDK client can point at — self-hosted, private, no per-token costs

It is a *deployment recipe*, not a framework: the weights come from
Hugging Face, and this repo is the tested glue (Docker image, serve flags,
RAM/tuning) that actually works on Blackwell — the part that usually takes
people days to get right (FlashInfer SM120 JIT, `cutlass-dsl`, graph-build
OOM).

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
import os
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8020/v1", api_key=os.environ["VLLM_API_KEY"])
resp = client.chat.completions.create(
    model="qwen3.8-27b",
    messages=[{"role": "user", "content": "Hello!"}],
)
```

## First boot & troubleshooting

Expect a slow **first** start (subsequent starts are much faster):

- image build: vLLM + FlashInfer wheels
- container boot: FlashInfer JIT-compiles the Blackwell FP4 GEMMs (`nvcc`
  inside the container — normal), then CUDA graph capture (RAM-hungry)
- the endpoint is ready once the logs show `Application startup complete`:

```bash
docker logs -f vllm
```

| Symptom | Likely cause / fix |
|---|---|
| stuck in "Compiling kernels…" / `nvcc` | normal first boot (JIT); skipped on later boots |
| OOM crash during graph capture | lower `MAX_JOBS` / `NVCC_THREADS` (e.g. `2`) |
| port 8020 in use | change the host port in `docker-compose.yml` `ports` |
| "model folder … does not exist" | `setup.sh` missing/incomplete — weights absent in `./models/` |
| HTTP 401 from the API | `VLLM_API_KEY` set in `.env` — send it as `Authorization: Bearer …` |

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

## Switching the model

The Docker image only ships vLLM — no model is baked in. Both `setup.sh` and
`docker compose` read the `.env` keys below; unset keys keep the defaults for
the stock Qwen3.8-27B NVFP4 model:

| Key | Meaning | Default |
|---|---|---|
| `MODEL_REPO` | Hugging Face repo to download | `gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090` |
| `MODEL_SUBDIR` | local weights dir under `./models/` (and the container mount) | `qwen3.8-27b-nvfp4` |
| `SERVED_MODEL_NAME` | value for `--served-model-name` | `qwen3.8-27b` |
| `CONTAINER_NAME` | container name | `vllm` |

To serve a different model:

1. set `MODEL_REPO` + `MODEL_SUBDIR` (+ `SERVED_MODEL_NAME`) in `.env`
2. `./setup.sh` — downloads the weights into `./models/$MODEL_SUBDIR/`
3. `docker compose up -d --build`

Model-specific serve flags (`--quantization`, `--kv-cache-dtype`, parsers, …)
deliberately stay explicit in `docker-compose.yml`. If your model needs
different ones, provide a full replacement of the `command` list in a second
file, e.g. `mymodel.yml`, and run
`docker compose -f docker-compose.yml -f mymodel.yml up -d`.

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
