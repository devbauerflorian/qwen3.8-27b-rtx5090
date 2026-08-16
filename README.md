# Qwen3.8-27B (NVFP4) on a single RTX 5090 with vLLM

Self-contained Docker deployment of
[**Qwen3.8-27B**](https://huggingface.co/Qwen/Qwen3.8-27B) in
[NVFP4 4-bit quantization](https://huggingface.co/gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090),
served by [vLLM](https://github.com/vllm-project/vllm) with an OpenAI-compatible
API — one 32 GB card, 256K-token context, reasoning and auto tool choice
included.

## Purpose

Who is this for, and why? This repo makes one specific thing work, end to
end, on a **single consumer GPU (RTX 5090, 32 GB)**:

- a **27B model** (Qwen3.8-27B) at **NVFP4 4-bit** — ~18.8 GB in VRAM
- the **native 256K-token context** (FP8 KV cache) — most 4-bit recipes cap
  out far earlier on 32 GB
- an **OpenAI-compatible API** with reasoning + auto tool choice (qwen3
  reasoning parser, qwen3_xml tool-call parser) that any OpenAI-SDK client can
  point at — self-hosted, private, no per-token costs

It is a *deployment recipe*, not a framework: the weights come from
Hugging Face, and this repo is the tested glue (Docker image, serve flags,
RAM/tuning) that actually works on Blackwell — the part that usually takes
people days to get right (FlashInfer SM120 JIT, `cutlass-dsl`, graph-build
OOM).

**Target host:** a plain **Ubuntu 24.04/26.04** box (bare metal or VM) — no
WSL2 or extra container layers in between. The host only needs Docker and a
datacenter-branch NVIDIA driver; `setup.sh` installs the rest (NVIDIA
container toolkit), and the image ships its own CUDA toolkit. Once
`docker compose up -d` is up, `8020` (direct full API) works immediately;
`8030` (Basic-auth gateway) serves once `METRICS_HASH` is in `.env`
(fail-closed by design — see Authentication & exposure).

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
| Context | full 262,144 tokens fit in 32 GB (KV pool ≈ 265K tokens at the default `--gpu-memory-utilization 0.95`; ≈ 276K at `0.97`) |
| Hardware | Blackwell tensor cores only — Hopper can load the files but cannot run NVFP4 |
| License | Apache-2.0 (same as the base model) |

Numbers from the [model card](https://huggingface.co/gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090):
~80 tok/s single-stream decode, up to ~1,030 tok/s aggregate at 16 concurrent
requests, and a 5/5 tool-call pass rate. The card also benchmarks it against
the alternative [Unsloth NVFP4 export](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4):
where that one needs ~22.7 GB of weights and holds only a ~77K-token KV pool
(~42 tok/s decode, 2/5 tool calls), this one serves the full 256K context with
a ~276K-token KV pool (~80 tok/s decode, 5/5 tool calls) on the same card.

The weights are pulled into `./models/qwen3.8-27b-nvfp4/` by `setup.sh` and are
**not** part of this repository; this repo's tooling is MIT-licensed (see
`LICENSE.md`).

## Performance

Measured on this stack: 1x RTX 5090 (32 GB), driver `610.43.02`, vLLM 0.27.1,
NVFP4 weights (~18.8 GB) + FP8 KV cache, `--gpu-memory-utilization 0.95`
(KV pool ≈ 265K tokens). Single stream, sequential requests, `temperature 0`,
128-token completion budget, and a unique padding per run — i.e. a cold
prefill: the stack runs with `--enable-prefix-caching`, and any repeat of a
seen prefix hits the cache instead. Re-measure with:

```bash
.venv_download/bin/python benchmarks/context_bench.py --cached
```

(the suite reads `VLLM_API_KEY` from `.env`; sizes/runs are env-configurable,
see the script header. Raw runs land in `benchmarks/results/`).

| context | prompt (tok) | TTFT (s) | prefill (tok/s) | decode (tok/s) | E2E (s) |
|---|---|---|---|---|---|
| 1K | 1,086 | 0.07 | 15,785 | 81.5 | 0.58 |
| 8K | 8,254 | 0.59 | 14,085 | 80.3 | 1.17 |
| 32K | 32,830 | 3.36 | 9,773 | 78.0 | 3.92 |
| 64K | 65,598 | 9.58 | 6,845 | 74.9 | 10.23 |
| 128K | 131,133 | 30.95 | 4,237 | 69.5 | 31.67 |
| 250K | 256,062 | 104.18 | 2,458 | 61.6 | 104.96 |

Medians of 3 timed runs per size (2026-08-16, `benchmarks/results/`).
Prefill throughput falls with context length (quadratic attention + KV
writes), while decode holds ~80 tok/s up to 32K context and drifts to ~62 at
the 250K row, where the big KV cache dominates each step. The 250K row is
256,062 prompt tokens (≈98% of the 262,144 `--max-model-len` cap; the
completion budget keeps the request inside the ~265K-token KV pool at the
default `0.95` utilization).

With the prefix cache warmed (same padding repeated — the `--cached` flag of
the suite), only the unseen suffix is prefilled:

| context | cold TTFT (s) | cached TTFT (s) |
|---|---|---|
| 1K | 0.07 | 0.07 |
| 8K | 0.59 | 0.05 |
| 32K | 3.36 | 0.23 |
| 64K | 9.58 | 0.30 |
| 128K | 30.95 | 0.51 |
| 250K | 104.18 | 0.65 |

Measured on a shared host: other GPU workloads (including the agent sessions
served by this very stack) can shift the numbers a bit. For aggregate
throughput under parallel load, run the companion suite
`benchmarks/parallel_bench.py` — concurrent-request runs (aggregate tok/s vs
concurrency) to re-measure the ~1,030 tok/s @ 16 figure above on your host.

## Requirements

| | |
|---|---|
| GPU | 1x NVIDIA RTX 5090 (32 GB VRAM) |
| System RAM | 64 GB minimum, 128 GB recommended (CUDA graph builds are RAM-hungry) |
| Host | Ubuntu 24.04/26.04, Docker, NVIDIA driver (datacenter branch) |
| Disk | ~20 GB for the model weights, ~10 GB for the Docker image |

The container is fully self-contained (it ships its own CUDA toolkit). On the
host you only need the NVIDIA driver and the NVIDIA container toolkit, which
`setup.sh` installs.

## Quickstart

```bash
# 1. Install the NVIDIA container toolkit and download the model (~20 GB)
#    (public model, no HF_TOKEN needed)
sudo ./setup.sh

# 2. Configure
cp .env.example .env
# edit .env:
#   - VLLM_API_KEY: set it, or leave empty to disable auth (LAN use only!)
#   - 8030 gateway (optional): METRICS_USER / METRICS_PASSWORD + METRICS_HASH
#     (bcrypt one-liner in .env.example) — caddy stays down without the hash

# 3. Build and start (first build takes a while)
docker compose up -d --build

# 4. Check it serves (open when VLLM_API_KEY is empty; with a key set,
#    add -H "Authorization: Bearer $VLLM_API_KEY")
curl http://localhost:8020/v1/models
```

Requests use the model name `qwen3.8-27b` (the compose default for
`--served-model-name`). Port and auth semantics of 8020/8030: see
[Authentication & exposure](#authentication--exposure) below.

### Using the API

```bash
# Direct LLM endpoint (auth per "Authentication & exposure")
curl http://localhost:8020/v1/chat/completions \
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

# Direct endpoint: vLLM validates the bearer key (VLLM_API_KEY). When no
# key is configured, any non-empty placeholder works.
client = OpenAI(
    base_url="http://localhost:8020/v1",
    api_key=os.environ.get("VLLM_API_KEY") or "no-key",
)
# … or via the Basic-auth gateway on 8030 (for externally-exposed setups):
# cred = base64.b64encode(
#     f"{os.environ['METRICS_USER']}:{os.environ['METRICS_PASSWORD']}".encode()
# ).decode()
# client = OpenAI(
#     base_url="http://localhost:8030/v1",
#     api_key="gateway",
#     default_headers={"Authorization": f"Basic {cred}"},
# )
resp = client.chat.completions.create(
    model="qwen3.8-27b",
    messages=[{"role": "user", "content": "Hello!"}],
)
```

## Authentication & exposure

The stack has four network surfaces (three reachable externally, one
loopback-only):

| Endpoint | What | Authentication |
|---|---|---|
| `:8020` | direct full vLLM API — OpenAI-compatible `/v1/*` plus `/metrics`, `/health`, `/docs` | `VLLM_API_KEY` empty → **fully open (LAN-use only!)**; key set → bearer on `/v1/*`, while `/metrics`, `/health`, `/docs` stay open by vLLM design |
| `:8030` | caddy gateway — routes `/v1/*` (translates Basic auth into the vLLM bearer key upstream), `/metrics` (vLLM telemetry), `/dcgm/metrics` (per-GPU DCGM telemetry) | always HTTP Basic (`METRICS_USER` / `METRICS_PASSWORD` from `.env`) |
| `:3000` | Grafana UI | open on all interfaces; protected by the Grafana admin password (sign-ups disabled) |
| `127.0.0.1:9090` | Prometheus (30 d retention) | loopback only — from a remote host, open an `ssh -L 9090:localhost:9090` tunnel first |

Notes:

- **Fresh checkout:** caddy stays down until `METRICS_HASH` (the bcrypt hash
  of `METRICS_PASSWORD`, see `.env.example`) is generated in `.env` —
  fail-closed, so the 8030 gateway is unavailable in the meantime.
- **Prometheus:** the internal scrape targets (on the docker network) are
  unauthenticated — only the host-side ports matter for external exposure.
- **Internet-facing box:** put 8020/8030/3000 behind a firewall or Tailnet;
  nothing behind 8030 is exposed unauthenticated, but 8020/3000 can be.

```bash
# /v1/* on 8020 — direct API (open without VLLM_API_KEY)
curl http://localhost:8020/v1/models

# … or the Basic-auth gateway on 8030 (vLLM + per-GPU telemetry)
curl -u "$METRICS_USER:$METRICS_PASSWORD" http://localhost:8030/v1/models
curl -u "$METRICS_USER:$METRICS_PASSWORD" http://localhost:8030/metrics | grep -m1 vllm:generation_tokens
curl -u "$METRICS_USER:$METRICS_PASSWORD" http://localhost:8030/dcgm/metrics | grep -m1 DCGM_FI_DEV_GPU_UTIL
ssh -L 9090:localhost:9090 <host>   # then open http://localhost:9090
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
| port 8020/8030 in use | set `VLLM_HOST_PORT` / `GATEWAY_HOST_PORT` in `.env` (defaults `8020` / `8030`) and re-run `docker compose up -d`; only touch the `ports` in `docker-compose.yml` (and the site line in `caddy/Caddyfile`) if you also want to move the caddy container port |
| "model folder … does not exist" | `setup.sh` missing/incomplete — weights absent in `./models/` |
| HTTP 401 from the API | see [Authentication & exposure](#authentication--exposure) — `8020` requires the `VLLM_API_KEY` bearer when set; `8030` requires Basic auth instead |
| `/metrics` open on `:8020` | by design (vLLM only keys `/v1/*`); the 8030 gateway routes are always behind Basic auth — block 8020/8030 in your firewall if you want no external exposure at all |

**Desktop host — keep the display off the RTX 5090.** Run X/Wayland (the
compositor) on your iGPU if your CPU has one (laptops: "integrated only"
graphics mode in the BIOS/firmware; desktops: point the display connector at
the iGPU) and leave the RTX 5090 a pure compute card. The display server no
longer takes its share of the 32 GB, so the VRAM is freed up for normal
operations and other tasks — first and foremost vLLM's KV cache and whatever
else you run on that card (in the test environment a GNOME shell running on
the 5090 had ~6 MiB of it booked; every GUI app sitting on the dGPU adds to
that).

## Monitoring (Prometheus + Grafana)

The compose stack includes a small monitoring sidecar set — it just comes up
with `docker compose up -d`:

| Service | Port | What it does |
|---|---|---|
| `vllm` | `8020` | direct full API access (auth per [Authentication & exposure](#authentication--exposure)); caddy (8030) and Prometheus also reach it on the docker network (`vllm:8000`) |
| `caddy` | `8030` | optional Basic-auth gateway (expose it to host the OpenAI endpoint externally) |
| `dcgm-exporter` | — | per-GPU DCGM telemetry sidecar (details in the paragraph below) |
| `prometheus` | `127.0.0.1:9090` | scrapes `vllm:8000/metrics` + `dcgm-exporter:9400` every 15 s; 30 d retention, loopback-only (remote access via SSH tunnel) |
| `grafana` | `3000` | dashboard auto-provisioned under folder **vllm** ("vLLM — Qwen3.8-27B (RTX5090)"): running/waiting requests, token throughput, TTFT / E2E / inter-token latency, KV-cache utilization, preemptions, prefix-cache hit ratio |

Grafana login: `admin` / `GRAFANA_ADMIN_PASSWORD` from `.env`.

The dashboard is reachable at these URLs (Grafana listens on all
interfaces, port 3000 — from another machine on the network, replace
`localhost` with the host's IP/DNS name):

- `http://localhost:3000/` — the Grafana UI (the dashboard sits in folder
  **vllm**, see the table above)
- `http://localhost:3000/d/vllm-qwen3827b` — direct link to the dashboard

![Top rows of the auto-provisioned vLLM dashboard](docs/Dashboard_Example.png)

*Top rows of the dashboard (folder **vllm**): requests running/waiting,
KV-cache utilization, TTFT p95, token counters, requests per second.*

Note the caddy gateway on 8030 does *not* route Grafana — the dashboard
is only available on 3000. To run the queries behind the panels
yourself, hit Prometheus on the loopback (`http://localhost:9090`;
remote access via the SSH tunnel described in
[Authentication & exposure](#authentication--exposure)).

Per-GPU metrics (utilization, memory, power, temps, fan, clock) are part of
the stack: the `dcgm-exporter` service (no host port) plus the `dcgm`
Prometheus job (scrapes `dcgm-exporter:9400`). The exporter is a small NVML
sidecar built from public images (`dcgm-exporter/`) that emits the official
`DCGM_FI_DEV_*` names — no nvcr.io/NGC login required; the NVML driver
library is injected into the container by the NVIDIA container toolkit.
Read them through the gateway at `http://localhost:8030/dcgm/metrics` or
query the `DCGM_*` series in Grafana Explore. Prefer the official exporter?
Replace the dcgm-exporter `build:` block in `docker-compose.yml` with
`image: nvcr.io/nvidia/dcgm-exporter:3.3.9-3.6.0-ubuntu22.04`. Not needed?
Delete the service + the `dcgm` job and `docker compose up -d`.

## Configuration

Most knobs live in `docker-compose.yml`:

| Setting | Value | Notes |
|---|---|---|
| LLM port | `8020` | direct full vLLM API host port; override with `VLLM_HOST_PORT` in `.env`, unset = `8020` (auth semantics: see [Authentication & exposure](#authentication--exposure)) |
| Gateway port | `8030` | caddy host port (Basic-auth front, for optional external exposure); override with `GATEWAY_HOST_PORT` in `.env`, unset = `8030`; container-side ports (`vllm:8000`, `caddy:8030`) stay fixed |
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
| dcgm-exporter (GPU metrics sidecar) | `python:3.13-slim` + `nvidia-ml-py 13.610.43` | `dcgm-exporter/Dockerfile` |
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
requirements.lock                   frozen Python stack of the known-good container (196 packages)
docker-compose.yml                  service definition (ports, volumes, serve flags)
setup.sh                            host prerequisites + model download (~20 GB)
.env.example                        secret template (copy to .env)
benchmarks/context_bench.py         context-size performance suite (TTFT / prefill / decode per size)
benchmarks/parallel_bench.py        concurrent-requests throughput suite (aggregate tok/s vs concurrency)
# raw runs of both suites land in benchmarks/results/
caddy/Caddyfile                     Basic-auth gateway: /v1/*, /metrics, /dcgm/metrics (port 8030)
prometheus/prometheus.yml           Prometheus scrape config (vllm + dcgm jobs)
grafana/                            auto-provisioned datasource + vLLM dashboard
dcgm-exporter/                      NVML → Prometheus sidecar (DCGM_FI_DEV_* names, no NGC login)
models/qwen3.8-27b-nvfp4/           model weights (empty in git, filled by setup.sh)
```
