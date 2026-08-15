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

# 4. Check it serves (401 without Basic auth is expected)
curl -u "$METRICS_USER:$METRICS_PASSWORD" http://localhost:8020/v1/models
```

The API is exposed on `http://localhost:8020` through the caddy gateway
(HTTP Basic auth: `METRICS_USER` / `METRICS_PASSWORD` from `.env`; caddy
translates it into the vLLM bearer key upstream). vLLM itself is
internal-only (container port 8000, docker network) and is served under the
model name `qwen3.8-27b`.

### Using the API

```bash
curl -u "$METRICS_USER:$METRICS_PASSWORD" http://localhost:8020/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.8-27b",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 256
  }'
```

Any OpenAI SDK works, e.g. with Python:

```python
import base64, os
from openai import OpenAI

# The 8020 endpoint is fronted by the caddy gateway: it speaks HTTP Basic
# auth and forwards to vLLM with its own bearer key. `api_key` just needs
# to be non-empty — authentication happens via the Authorization header.
cred = base64.b64encode(
    f"{os.environ['METRICS_USER']}:{os.environ['METRICS_PASSWORD']}".encode()
).decode()
client = OpenAI(
    base_url="http://localhost:8020/v1",
    api_key="caddy-gateway",
    default_headers={"Authorization": f"Basic {cred}"},
)
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
| port 8020 in use | change the caddy host port in `docker-compose.yml` `ports` (and the site line in `caddy/Caddyfile` if you move the container port too) |
| "model folder … does not exist" | `setup.sh` missing/incomplete — weights absent in `./models/` |
| HTTP 401 from the API | the caddy gateway requires Basic auth — send `METRICS_USER` / `METRICS_PASSWORD` from `.env` (`curl -u user:pass`) |
| Basic auth on `:8020/…` | use `METRICS_USER` / `METRICS_PASSWORD` from `.env` (caddy gateway) |
| `:8020` no longer serves `/metrics` open | by design — every caddy route requires Basic auth; Prometheus scrapes `vllm:8000` + `dcgm-exporter:9400` on the docker network instead; block 8020 in your firewall if you want no external exposure at all |

## Monitoring (Prometheus + Grafana)

The compose stack includes a small monitoring sidecar set — it just comes up
with `docker compose up -d`:

| Service | Port | What it does |
|---|---|---|
| `vllm` | — | internal-only: caddy and Prometheus reach it on the docker network (`vllm:8000`); the bearer key `VLLM_API_KEY` is what caddy sends upstream |
| `caddy` | `8020` | Basic-auth gateway on the old vLLM port: routes `/v1/*` (translates Basic auth into the vLLM bearer token), `/metrics` (vLLM telemetry) and `/dcgm/metrics` (per-GPU telemetry); `METRICS_USER` / `METRICS_PASSWORD` from `.env` |
| `dcgm-exporter` | — | DCGM exporter (util, memory, power, temps); public access via `8020/dcgm/metrics`, no host port |
| `prometheus` | `127.0.0.1:9090` | scrapes `vllm:8000/metrics` + `dcgm-exporter:9400` every 15 s, 30 d retention (host-local only — reach it via SSH tunnel if needed) |
| `grafana` | `3000` | dashboard auto-provisioned under folder **vllm** ("vLLM — Qwen3.8-27B (RTX5090)"): running/waiting requests, token throughput, TTFT / E2E / inter-token latency, KV-cache utilization, preemptions, prefix-cache hit ratio |

Grafana login: `admin` / `GRAFANA_ADMIN_PASSWORD` from `.env`.

Exposure posture: Grafana is open on all interfaces (3000) and protected by
the `.env` admin password (sign-ups disabled); Prometheus stays loopback-only
(9090 — SSH tunnel). The only open API port is 8020 (the caddy gateway) and
every route behind it is Basic-authed, so nothing is exposed unauthenticated.
If this box faces the internet, put 8020/3000 behind a firewall or Tailnet.
On a fresh checkout `caddy` stays down until `METRICS_HASH` is generated in
`.env` (fail-closed — nothing is exposed unauthenticated in the meantime).

```bash
# /v1/* via the Basic-auth gateway on 8020
curl -u "$METRICS_USER:$METRICS_PASSWORD" http://localhost:8020/v1/models

# vLLM + per-GPU telemetry through the gateway (Basic auth)
curl -u "$METRICS_USER:$METRICS_PASSWORD" http://localhost:8020/metrics | grep -m1 vllm:generation_tokens
curl -u "$METRICS_USER:$METRICS_PASSWORD" http://localhost:8020/dcgm/metrics | grep -m1 DCGM_FI_DEV_GPU_UTIL
ssh -L 9090:localhost:9090 <host>   # then open http://localhost:9090
```

Per-GPU metrics (DCGM: utilization, memory, power, temps) are part of the
stack: the `dcgm-exporter` service (no host port) plus the `dcgm` Prometheus
job (scrapes `dcgm-exporter:9400`). Read them through the gateway at
`http://localhost:8020/dcgm/metrics` or query the `DCGM_*` series in Grafana
Explore. Not needed? Delete the service + the `dcgm` job and
`docker compose up -d`.

## Configuration

Most knobs live in `docker-compose.yml`:

| Setting | Value | Notes |
|---|---|---|
| Gateway port | `8020` | caddy host port (was the direct vLLM port); vLLM container port `8000` is internal-only (docker network) |
| `VLLM_API_KEY` | from `.env` | empty = no authentication |
| GPU | 1x, `CUDA_VISIBLE_DEVICES=0` | single RTX 5090 |
| `MAX_JOBS` / `NVCC_THREADS` | `4` | lower these on machines with less RAM |
| `--gpu-memory-utilization` | `0.95` | upstream card benchmarks at `0.97` (KV pool ≈ 276K tokens, full 256K); `0.90` holds only ~205K |
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
| `MODEL_REVISION` | Hugging Face revision to download (weights pin, `setup.sh` only) | `69274a0d…` (known-good, 2026-08-15) |

Set `MODEL_REVISION=` (empty) in `.env` if you rather want `setup.sh` to
always download the latest weights.

To serve a different model:

1. set `MODEL_REPO` + `MODEL_SUBDIR` (+ `SERVED_MODEL_NAME`) in `.env`
2. `./setup.sh` — downloads the weights into `./models/$MODEL_SUBDIR/`
3. `docker compose up -d --build`

Model-specific serve flags (`--quantization`, `--kv-cache-dtype`, parsers, …)
deliberately stay explicit in `docker-compose.yml`. If your model needs
different ones, provide a full replacement of the `command` list in a second
file, e.g. `mymodel.yml`, and run
`docker compose -f docker-compose.yml -f mymodel.yml up -d`.

## Pinned versions

Everything that could break a rebuild is pinned — the Python stack is the
full lock of the known-good production container (frozen 2026-08-15):

| Input | Pinned | Where |
|---|---|---|
| vLLM stack (vllm 0.27.1, FlashInfer, CUTLASS DSL, torch, … 196 packages) | `requirements.lock` | `Dockerfile` |
| uv / Python | `0.12.5` / `3.13.15` | `Dockerfile` |
| CUDA base image | `13.3.1-devel-ubuntu22.04` | `Dockerfile` |
| huggingface_hub (model download CLI) | `1.27.0` | `setup.sh` |
| NVIDIA container toolkit | `1.20.0-1` | `setup.sh` |
| Model weights | HF revision `69274a0d…` | `setup.sh` (override: `MODEL_REVISION`) |
| Host driver (tested) | `610.43.02` (RTX 5090) | — |
| caddy (API gateway, monitoring) | `2.11.4-alpine` | `docker-compose.yml` |
| dcgm-exporter (per-GPU metrics) | `3.3.9-3.6.0-ubuntu22.04` | `docker-compose.yml` |
| prometheus (metrics) | `v3.13.2` | `docker-compose.yml` |
| grafana (dashboards) | `11.1.4` | `docker-compose.yml` |

Updating the pins: change the relevant lines, rebuild the image, re-run the
smoke test (`curl /v1/models` + a short chat), and commit. For a new
stack, regenerate the lock from a working container:
`docker exec <container> sh -c 'cd /app && uv pip freeze' > requirements.lock`
(keep only the `package==version` lines).

## Project layout

```
Dockerfile                          vLLM + flashinfer + CUTLASS DSL image (NVFP4)
docker-compose.yml                  service definition (ports, volumes, serve flags)
setup.sh                            host prerequisites + model download (~20 GB)
.env.example                        secret template (copy to .env)
caddy/Caddyfile                   Basic-auth gateway: /v1/*, /metrics, /dcgm/metrics (port 8020)
prometheus/prometheus.yml          Prometheus scrape config (vllm + dcgm jobs)
grafana/                           auto-provisioned datasource + vLLM dashboard
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
**not** part of this repository; this repo's tooling is MIT-licensed (see `LICENSE.md`).
