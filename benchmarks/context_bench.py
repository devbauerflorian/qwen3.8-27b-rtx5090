#!/usr/bin/env python3
"""Context-size benchmark for the Qwen3.8-27B (NVFP4) vLLM deployment.

Measures single-stream performance across context sizes through the
OpenAI-compatible /v1/chat/completions endpoint:

    context size -> TTFT, prefill throughput, decode throughput, E2E latency

The suite runs sequentially (one request at a time):
  1. one warm-up request per context size (absorbs first-call effects),
  2. N timed requests with *different* padding text per run — the server is
     started with --enable-prefix-caching (see docker-compose.yml), so a
     repeated prefix would turn a "cold prefill" into a cache hit and
     inflate the prefill numbers.

Usage (Python >= 3.9; needs `httpx` plus optional `tokenizers` for exact
token-accurate padding — both live in .venv_download/ in this repo):

    .venv_download/bin/python benchmarks/context_bench.py [options]

Environment (every BENCH_* key is read from the process environment first,
then from the repo .env — process env wins):
  BENCH_BASE_URL    endpoint, default http://localhost:8020
  BENCH_API_KEY     bearer key (default: VLLM_API_KEY from .env)
  BENCH_BASIC_AUTH  "user:pass" — use instead of the API key (caddy
                     gateway on 8030: METRICS_USER / METRICS_PASSWORD)
  BENCH_MODEL       served model name (default: SERVED_MODEL_NAME from .env,
                     falling back to qwen3.8-27b)
  BENCH_SIZES       comma list, default 1024,8192,32768,65536,131072,256000
  BENCH_RUNS        timed runs per size (default 3)
  BENCH_MAX_TOKENS  completion budget in tokens (default 128)
  BENCH_TOKENIZER   model tokenizer.json (default: auto-detected under ./models/)
  BENCH_TIMEOUT_S   per-request timeout (default 600)

CLI flags:
  --cached      additionally run one request that repeats the last timed
                seed per size, to show the TTFT delta of the server-side
                prefix cache
  --quick       only the two smallest sizes, 1 run each (sanity check)

Results are written to benchmarks/results/<timestamp>.json (and
benchmarks/results/latest.json) and a paste-ready markdown table is printed
at the end. Individual run failures are recorded in the JSON (excluded from
the medians) instead of aborting the suite, and a partial report is written
on Ctrl-C.
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SIZES = [1024, 8192, 32768, 65536, 131072, 256000]
MAX_MODEL_LEN = 262144  # --max-model-len in docker-compose.yml

# Base instruction; the padded corpus is prepended to this.
# Fixed, controlled completion length (~50-100 tokens) so decode throughput
# is measurable at every context size.
INSTRUCTION = (
    "\n\nYou have now read a long passage of padding text. "
    "Do not summarize it. Reply with the word READY repeated exactly 50 times, "
    "separated by single spaces, and output nothing else."
)

# A deterministic, non-repetitive-feeling paragraph pool. Repeated far enough
# that the tokenized corpus covers the largest context size with room to
# rotate a per-run window (unique window start => no prefix-cache hits).
BASE_PARAGRAPHS = [
    "The deployment serves a twenty seven billion parameter model from a single "
    "consumer grade graphics card, using a four bit floating point weight format "
    "that NVIDIA introduced for the Blackwell generation of datacenter silicon.",
    "Prefill latency grows roughly linearly with the number of input tokens, while "
    "decode throughput is bounded by the memory bandwidth available to move the "
    "activations and key value cache of one sequence per step.",
    "The key value cache is stored in an eight bit floating point format here, which "
    "halves the video memory footprint relative to sixteen bit storage and is the "
    "reason a half million token pool fits alongside the quantized weights.",
    "A shared prefix cache lets the engine skip the already computed layers of any "
    "input that was seen recently, which makes repeated system prompts cheap for "
    "serving many related conversations on the same machine.",
    "The gateway in front of the engine speaks plain HTTP basic authentication on a "
    "separate host port and forwards to the model server using a bearer token kept "
    "only in the process environment of the stack.",
]
# ~4.1M chars ≈ 1.1M tokens — comfortably more than the largest context
# target (256K), so the per-run rotating window can never be truncated.
CORPUS = " ".join(BASE_PARAGRAPHS * 3000)


def load_env_file(path: Path) -> dict:
    """Read KEY=VALUE pairs from .env (unquoted values, lines only)."""
    values = {}
    if not path.exists():
        return values
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        values[key.strip()] = val.strip().strip('"')
    return values


def detect_tokenizer() -> str | None:
    """Auto-detect the first ./models/*/tokenizer.json (sorted) if present.

    A missing models/ directory is not an error — the char-heuristic padder
    takes over (see make_padder).
    """
    models = REPO_ROOT / "models"
    if not models.is_dir():
        return None
    cands = sorted(models.glob("*/tokenizer.json"))
    return str(cands[0]) if cands else None


def make_padder(tokenizer_path: str | None):
    """Return padder(n_tokens, seed) -> padding text.

    With a tokenizer: exactly n_tokens (rotating window over a tokenized
    corpus, so different seeds yield different token sequences). Without:
    a character heuristic (~4.1 chars/token for this model) that the
    server will correct via usage.prompt_tokens.
    """
    if tokenizer_path:
        from tokenizers import Tokenizer

        tok = Tokenizer.from_file(tokenizer_path)
        corpus_ids = tok.encode(CORPUS).ids
        window_size = len(corpus_ids)
        doubled = corpus_ids * 2  # wrap-around source; list copy, shared refs

        def padder(n: int, seed: int) -> str:
            if n > window_size:
                raise SystemExit(
                    f"padder corpus too small: {window_size} ids < {n} target — "
                    f"extend CORPUS in context_bench.py")
            start = (seed * 7919) % window_size
            # n unique tokens from a seed-rotated window: different seed =>
            # different window start => no prefix-cache hits. decode()
            # re-serializes the ids; the server re-tokenizes them back to the
            # (almost) same sequence — a few tokens of drift is tolerable.
            ids = list(itertools.islice(doubled, start, start + n))
            return tok.decode(ids)

        return padder

    def padder(n: int, seed: int) -> str:
        # heuristic: ~4.1 chars per token for Qwen3 on English padding
        chars = int(n * 4.1)
        chunk = CORPUS * ((chars // len(CORPUS)) + 1)
        # rotate by seed so runs differ
        offset = (seed * 1331) % len(chunk)
        return (chunk[offset:] + chunk[:offset])[:chars]

    return padder


@dataclass
class RunResult:
    size_label: str
    target_tokens: int
    prompt_tokens: int
    completion_tokens: int
    ttft_s: float
    prefill_tps: float
    decode_tps: float
    itl_median_ms: float
    e2e_s: float
    e2e_tps: float
    seed: int
    cached: bool = False
    note: str = ""

    def is_error(self) -> bool:
        return self.note.startswith("error:")


def _sse_urllib(url: str, headers: dict, payload: dict, timeout_s: float):
    """Stdlib fallback SSE reader, used only when httpx is unavailable.

    Yields like http_stream_request and raises the same
    RuntimeError("HTTP <code>: ...") for non-2xx responses.
    """
    import urllib.error
    import urllib.request

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            for raw in resp:
                line = raw.decode(errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                yield data, time.perf_counter() - t0
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body[:300]}")


def http_stream_request(base_url: str, headers: dict, payload: dict, timeout_s: float):
    """POST /v1/chat/completions (SSE), yielding (event_json, elapsed_s).

    elapsed is measured from the instant the request is sent, so the first
    yielded item approximates TTFT on both transports. Non-2xx responses
    raise RuntimeError("HTTP <code>: <body head>") on both transports,
    which run_once's main loop catches for the thinking-off retry.
    """
    url = base_url.rstrip("/") + "/v1/chat/completions"
    try:
        import httpx
    except ImportError:
        yield from _sse_urllib(url, headers, payload, timeout_s)
        return

    with httpx.Client(timeout=httpx.Timeout(timeout_s)) as client:
        t0 = time.perf_counter()
        with client.stream("POST", url, json=payload, headers=headers) as resp:
            if resp.status_code != 200:
                body = resp.read().decode(errors="replace")
                raise RuntimeError(f"HTTP {resp.status_code}: {body[:300]}")
            for line in resp.iter_lines():
                line = line.strip()
                if not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                yield data, time.perf_counter() - t0


def run_once(base_url: str, headers: dict, model: str, padder, target_tokens: int,
             seed: int, max_tokens: int, timeout_s: float,
             use_thinking_off: bool = True) -> dict:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": f"[bench seed {seed}] {padder(target_tokens, seed)}"
                + INSTRUCTION,
            }
        ],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if use_thinking_off:
        payload["chat_template_kwargs"] = {"enable_thinking": False}

    content_ts = []
    last_data = None
    usage = {}
    for data, now in http_stream_request(base_url, headers, payload, timeout_s):
        parsed = json.loads(data)
        last_data = parsed
        if "usage" in parsed and parsed["usage"]:
            usage = parsed["usage"]
        choice = (parsed.get("choices") or [{}])[0]
        delta = choice.get("delta") or {}
        piece = delta.get("content") or delta.get("reasoning_content")
        if piece:
            content_ts.append(now)  # first element == TTFT (relative to t0)

    if last_data is None:
        raise RuntimeError("no SSE data received")
    if usage:
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
    else:
        # fallback estimate: prompt from our own token count is unreliable;
        # mark it for a warning
        prompt_tokens = -1
        completion_tokens = len(content_ts)

    ttft = content_ts[0] if content_ts else 0.0
    e2e = content_ts[-1] if content_ts else ttft
    n_tok = max(completion_tokens, len(content_ts), 1)
    inter = [b - a for a, b in zip(content_ts, content_ts[1:])]
    decode_tps = 0.0
    if e2e - ttft > 1e-6 and n_tok > 1:
        decode_tps = (n_tok - 1) / (e2e - ttft)
    prefill_tps = (prompt_tokens / ttft) if ttft > 1e-6 and prompt_tokens > 0 else 0.0
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "ttft_s": ttft,
        "prefill_tps": prefill_tps,
        "decode_tps": decode_tps,
        "itl_median_ms": (statistics.median(inter) * 1000) if inter else 0.0,
        "e2e_s": e2e,
        "e2e_tps": ((max(prompt_tokens, 0) + n_tok) / e2e) if e2e > 1e-6 else 0.0,
    }


def gpu_snapshot() -> dict:
    """Snapshot GPU 0 for the report; {} if nvidia-smi is unavailable or
    the output does not parse (never fails the bench)."""
    if not shutil.which("nvidia-smi"):
        return {}
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,power.draw,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        parts = [p.strip() for p in out.stdout.strip().splitlines()[0].split(",")]
        if len(parts) != 5:  # multi-GPU output or unexpected format
            return {}
        name, util, mem, power, temp = parts
        return {"name": name, "util_pct": int(util), "mem_used": f"{mem} MiB",
                "power_w": float(power), "temp_c": int(temp)}
    except Exception:
        return {}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cached", action="store_true",
                    help="add one repeated-seed run per size (prefix-cache TTFT delta)")
    ap.add_argument("--quick", action="store_true",
                    help="only the two smallest sizes, 1 timed run (sanity check)")
    args = ap.parse_args()

    env_file = load_env_file(REPO_ROOT / ".env")

    def cfg(key: str, default: str | None = None) -> str | None:
        """Process env wins; then the repo .env; then the default."""
        return os.environ.get(key) or env_file.get(key) or default

    base_url = cfg("BENCH_BASE_URL", "http://localhost:8020")
    basic = cfg("BENCH_BASIC_AUTH")
    api_key = os.environ.get("BENCH_API_KEY") or env_file.get("VLLM_API_KEY", "")
    model = cfg("BENCH_MODEL", env_file.get("SERVED_MODEL_NAME") or "qwen3.8-27b")
    timeout_s = float(cfg("BENCH_TIMEOUT_S", "600"))
    sizes_raw = cfg("BENCH_SIZES", ",".join(str(s) for s in DEFAULT_SIZES))
    try:
        sizes = [int(x.strip()) for x in sizes_raw.split(",") if x.strip()]
    except ValueError as exc:
        sys.exit(f"invalid BENCH_SIZES={sizes_raw!r} (want comma-separated "
                 f"integers): {exc}")
    if not sizes:
        sys.exit(f"no usable context sizes in BENCH_SIZES={sizes_raw!r}")
    runs = int(cfg("BENCH_RUNS", "3"))
    if runs < 1:
        runs = 1
    max_tokens = int(cfg("BENCH_MAX_TOKENS", "128"))
    if args.quick:
        sizes = sizes[:2]
        runs = 1

    headers = {}
    if basic:
        import base64
        user, _, pw = basic.partition(":")
        if not user or not pw:
            sys.exit(f"BENCH_BASIC_AUTH must be 'user:pass', got {basic!r}")
        headers["Authorization"] = "Basic " + base64.b64encode(
            f"{user}:{pw}".encode()).decode()
    elif api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if not headers:
        sys.exit("no credentials: set BENCH_API_KEY or BENCH_BASIC_AUTH (or .env)")

    header_note = "basic-auth (gateway)" if basic else "bearer api-key"
    tokenizer_path = cfg("BENCH_TOKENIZER") or detect_tokenizer()
    padder = make_padder(tokenizer_path)
    sizes_note = ("tokenizers not installed; padding sized by char heuristic"
                  if tokenizer_path is None else
                  "token-accurate padding via model tokenizer")

    # fit inside the served max model length (prompt + completion + template overhead)
    safety = 256
    ceiling = MAX_MODEL_LEN - max_tokens - safety
    sizes = sorted(set(min(s, ceiling) for s in sizes))

    def fmt_size(t: int) -> str:
        return f"{t // 1024}K" if t >= 1024 else str(t)

    print(f"base_url : {base_url} ({header_note})")
    print(f"model    : {model}")
    print(f"padding  : {sizes_note}")
    print(f"sizes    : {sizes} target tokens | max_tokens={max_tokens} | "
          f"timed runs/size={runs}")

    gpu_before = gpu_snapshot()
    if gpu_before:
        print(f"gpu      : {gpu_before['name']} | {gpu_before['util_pct']}% | "
              f"{gpu_before['mem_used']} | {gpu_before['power_w']} W | {gpu_before['temp_c']} C")

    results: list[RunResult] = []
    transport = "httpx" if importlib.util.find_spec("httpx") else "urllib"

    seed_base = int(time.time()) % 100000

    def fail_row(label: str, target: int, seed: int, cached: bool, exc: Exception) -> RunResult:
        msg = str(exc) or exc.__class__.__name__
        return RunResult(label, target, -1, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                         seed, cached, f"error: {msg}")

    try:
        for idx, target in enumerate(sizes):
            label = fmt_size(target)

            def execute(seed: int) -> dict:
                try:
                    return run_once(base_url, headers, model, padder, target, seed,
                                    max_tokens, timeout_s)
                except RuntimeError as exc:
                    if "HTTP 400" in str(exc):
                        # retry once without the thinking-off flag (older servers)
                        return run_once(base_url, headers, model, padder, target, seed,
                                        max_tokens, timeout_s, use_thinking_off=False)
                    raise

            # warm-up with a seed that is never reused in timed runs
            print(f"[{label}] warm-up ...", flush=True)
            try:
                execute(seed_base + 9001)
            except Exception as exc:
                # a 404 means the model name is wrong — stop instead of burning
                # the whole suite on guaranteed failures at every size
                if "HTTP 404" in str(exc):
                    sys.exit(f"[{label}] HTTP 404 from {base_url} — check BENCH_MODEL "
                             f"against the served model name. Detail: {exc}")
                print(f"[{label}] warm-up failed: {exc} (continuing)", flush=True)

            results_runs = []
            for r in range(runs):
                seed = seed_base + idx * 100 + r
                try:
                    m = execute(seed)
                except Exception as exc:
                    # transient/network failure: record it, keep the suite alive
                    print(f"[{label}] run {r + 1}/{runs} FAILED: {exc}", flush=True)
                    results_runs.append(fail_row(label, target, seed, False, exc))
                    continue
                note = ""
                if m["prompt_tokens"] < 0:
                    note = "prompt_tokens unavailable (no usage in stream)"
                elif abs(m["prompt_tokens"] - target) / target > 0.10:
                    # >10% drift only happens with the char-heuristic padder
                    note = f"prompt_tokens={m['prompt_tokens']} vs target {target}"
                print(f"[{label}] run {r + 1}/{runs}  "
                      f"ttft={m['ttft_s']:.3f}s prefill={m['prefill_tps']:8.1f} tok/s "
                      f"decode={m['decode_tps']:6.2f} tok/s e2e={m['e2e_s']:.2f}s "
                      f"(prompt {m['prompt_tokens']} tok)"
                      + (f"  [{note}]" if note else ""), flush=True)
                results_runs.append(RunResult(label, target, m["prompt_tokens"],
                                              m["completion_tokens"], m["ttft_s"],
                                              m["prefill_tps"], m["decode_tps"],
                                              m["itl_median_ms"], m["e2e_s"],
                                              m["e2e_tps"], seed, False, note))
            if args.cached:
                seed = seed_base + idx * 100 + (runs - 1)  # repeat the last timed seed
                print(f"[{label}] cached repeat ...", flush=True)
                try:
                    m = execute(seed)
                    print(f"[{label}] cached repeat: ttft={m['ttft_s']:.3f}s "
                          f"(prefix cache warmed by previous run)", flush=True)
                    results_runs.append(RunResult(label, target, m["prompt_tokens"],
                                                  m["completion_tokens"], m["ttft_s"],
                                                  m["prefill_tps"], m["decode_tps"],
                                                  m["itl_median_ms"], m["e2e_s"],
                                                  m["e2e_tps"], seed, True,
                                                  "cached prefix"))
                except Exception as exc:
                    print(f"[{label}] cached repeat FAILED: {exc}", flush=True)
                    results_runs.append(fail_row(label, target, seed, True, exc))
            results.extend(results_runs)
            # stop early when the endpoint rejects every request (e.g. 404 on a
            # bad model name discovered after a flaky warm-up)
            if results_runs and all(r.is_error() for r in results_runs):
                print(f"[{label}] all runs failed — {results_runs[0].note}", flush=True)
                break
    except KeyboardInterrupt:
        print("\ninterrupted — writing partial results", flush=True)

    gpu_after = gpu_snapshot()

    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "endpoint": base_url,
        "auth": "basic" if basic else "bearer",
        "model": model,
        "transport": transport,
        "max_tokens": max_tokens,
        "runs_per_size": runs,
        "cached_repeats": bool(args.cached),
        "notes": sizes_note,
        "gpu_before": gpu_before,
        "gpu_after": gpu_after,
        "runs": [asdict(r) for r in results],
    }
    out_dir = REPO_ROOT / "benchmarks" / "results"
    out_dir.mkdir(exist_ok=True)
    stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    for name in (f"{stamp}.json", "latest.json"):
        (out_dir / name).write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nresults: {out_dir / (stamp + '.json')}")

    def ok_runs(label_: str) -> list[RunResult]:
        return [r for r in results
                if r.size_label == label_ and not r.cached and not r.is_error()]

    def med(vals: list) -> float | None:
        return statistics.median(vals) if vals else None

    def cell(v: float | None, kind: str) -> str:
        if v is None:
            return "—"
        if kind == "int":
            return f"{max(v, 0):,.0f}"
        return f"{v:.1f}" if kind == "f1" else f"{v:.2f}"

    print("\n## Markdown (paste into README)\n")
    print("| context | prompt (tok) | TTFT (s) | prefill (tok/s) | decode (tok/s) | E2E (s) |")
    print("|---|---|---|---|---|---|")
    for target in sizes:
        label = fmt_size(target)
        okr = ok_runs(label)
        row = [label]
        prompts = [x.prompt_tokens for x in okr if x.prompt_tokens > 0]
        row.append(cell(med(prompts), "int"))
        row.append(cell(med([x.ttft_s for x in okr]), "f2"))
        row.append(cell(med([x.prefill_tps for x in okr]), "int"))
        row.append(cell(med([x.decode_tps for x in okr]), "f1"))
        row.append(cell(med([x.e2e_s for x in okr]), "f2"))
        print("| " + " | ".join(row) + " |")
    n_fail = sum(1 for r in results if r.is_error() and not r.cached)
    print("\nmedians of", runs, "timed runs per size, sequential single-stream "
          "(unique padding per run: cold prefill despite --enable-prefix-caching)"
          + (f"; {n_fail} failed run(s) excluded and recorded in the JSON"
             if n_fail else ""))
    if results and all(r.is_error() for r in results):
        sys.exit("all runs failed — no measurements to report")
    return 0


if __name__ == "__main__":
    sys.exit(main())
