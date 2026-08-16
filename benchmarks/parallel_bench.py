#!/usr/bin/env python3
"""Concurrent-requests throughput benchmark for the Qwen3.8-27B (NVFP4) vLLM deployment.

Measures how the stack scales with parallel load through the OpenAI-compatible
/v1/chat/completions endpoint:

    context size (default 8K, 32K, 64K, 128K) x concurrency level (1-16)
        -> aggregate output throughput, request rate, TTFT/E2E latencies,
           per-request decode speed

At each (context, level) combination, several batches are fired. Within a
batch, ALL requests start at the same instant (a threading barrier) and get
unique padding text — the server is started with --enable-prefix-caching (see
docker-compose.yml), so a repeated prefix would turn the timed prefill into a
cache hit. One warm-up batch per combination runs first and absorbs the
first-call effects.

Each combination is pool-capped: a level is skipped for a context when
level x (context + max_tokens) exceeds the ~265K-token KV pool at the
default --gpu-memory-utilization 0.95 (see README), so e.g. c=16 only runs
at 8K, c=8 up to 32K, c=4 up to 64K, c=2 up to 128K — no preemptions.

Usage (Python >= 3.9, same dependency set as context_bench.py — both live
in .venv_download/ in this repo):

    .venv_download/bin/python benchmarks/parallel_bench.py [options]

Environment (every BENCH_* key is read from the process environment first,
then from the repo .env — process env wins):
  BENCH_BASE_URL    endpoint, default http://localhost:8020
  BENCH_API_KEY     bearer key (default: VLLM_API_KEY from .env)
  BENCH_BASIC_AUTH  "user:pass" — use instead of the API key (caddy
                    gateway on 8030: METRICS_USER / METRICS_PASSWORD)
  BENCH_MODEL       served model name (default: SERVED_MODEL_NAME from .env,
                    falling back to qwen3.8-27b)
  BENCH_LEVELS      comma list of concurrency levels, default 1,2,4,8,16
  BENCH_CONTEXTS    comma list of prompt context sizes in tokens, default
                   8192,32768,65536,131072 — one (context, level) sweep per
                   size, pool-capped as described above
  BENCH_MAX_TOKENS  completion budget per request (default 256)
  BENCH_RUNS        timed batches per context x level combo (default 3)
  BENCH_WARMUP      warm-up batches per context x level combo (default 1)
  BENCH_TOKENIZER   model tokenizer.json (default: auto-detected under ./models/)
  BENCH_TIMEOUT_S   per-request timeout (default 600)

CLI flags:
  --quick   levels 1 and 8, 512-token context, 1 warm-up + 1 timed batch
            each (fast sanity check on an idle stack)

Per batch the suite records wall time, request rate, aggregate output and
total token throughput, TTFT/E2E percentiles, mean per-request decode speed,
and effective in-flight concurrency (Little's law: sum of the per-request
durations divided by batch wall time). Results are written to
benchmarks/results/<timestamp>_parallel.json (plus latest_parallel.json, so
context_bench.py's latest.json is left alone) and a paste-ready markdown
table is printed at the end.
"""

import argparse
import importlib.util
import json
import os
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from context_bench import (
    MAX_MODEL_LEN,
    REPO_ROOT,
    detect_tokenizer,
    gpu_snapshot,
    http_stream_request,
    load_env_file,
    make_padder,
)

DEFAULT_LEVELS = [1, 2, 4, 8, 16]
DEFAULT_CONTEXTS = [8192, 32768, 65536, 131072]
# Long controlled completion (READY x 250 ~ 250 tokens): with the default
# 256-token budget the model runs to the full budget, so the decode phase is
# long enough to measure sustained aggregate decode rather than a short tail
# after the prefill (context_bench's INSTRUCTION ends in ~43 tokens).
DEFAULT_INSTRUCTION = (
    "\n\nYou have now read a long passage of padding text. "
    "Do not summarize it. Reply with the word READY repeated exactly 250 times, "
    "separated by single spaces, and output nothing else."
)
DEFAULT_MAX_TOKENS = 256
# ~265K token KV pool at --gpu-memory-utilization 0.95 (see README).
KV_POOL_TOKENS_APPROX = 265000
# Seed offset for the 400 retry pass, so a retry never reuses the exact
# prefix its failed attempt may have parked in the prefix cache.
RETRY_SEED_OFFSET = 5000


def percentile(vals: list, p: float) -> float:
    """Linear-interpolated percentile (p in (0, 1)) of a list (0.0 if empty)."""
    if not vals:
        return 0.0
    vals = sorted(vals)
    if len(vals) == 1:
        return vals[0]
    k = (len(vals) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(vals) - 1)
    if lo == hi:
        return vals[lo]
    return vals[lo] + (vals[hi] - vals[lo]) * (k - lo)


def time_one(base_url, headers, model, padder, context, max_tokens, seed,
             timeout_s, thinking_off) -> dict:
    """Run one streaming chat completion, return per-request metrics.

    Same measurement scheme as context_bench.run_once (TTFT/E2E from SSE
    content-chunk timestamps, token counts from the final usage event) but
    with the longer DEFAULT_INSTRUCTION, so the decode phase stays long
    enough to measure sustained aggregate decode at high concurrency.
    """
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": f"[bench seed {seed}] {padder(context, seed)}"
                + DEFAULT_INSTRUCTION,
            }
        ],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if thinking_off:
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
        prompt_tokens = -1  # mark for a note in the report
        completion_tokens = len(content_ts)

    ttft = content_ts[0] if content_ts else 0.0
    e2e = content_ts[-1] if content_ts else ttft
    n_tok = max(completion_tokens, len(content_ts), 1)
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
        "e2e_s": e2e,
        "e2e_tps": ((max(prompt_tokens, 0) + n_tok) / e2e) if e2e > 1e-6 else 0.0,
    }


def fire_batch(base_url, headers, model, padder, context, max_tokens, seed_base,
               timeout_s, thinking_off, concurrency) -> tuple:
    """Fire `concurrency` requests at once. Returns (results, wall_s,
    thinking_off_used).

    All requests release a barrier first so they truly start together; wall
    time runs from dispatch to the last completion. Per-request errors are
    recorded in the result dict (not raised) so one bad request cannot sink
    the batch:

      * HTTP 404 (unknown model) is a guaranteed failure of the whole run —
        the caller must stop the suite (same as context_bench's warm-up).
      * HTTP 400 while the thinking-off flag is on means the server rejected
        chat_template_kwargs — the failed requests are fired once more
        without the flag (context_bench's fallback), with fresh seeds so the
        retry cannot hit the prefix its failed attempt parked in the cache.
    """
    state = {"thinking": thinking_off}

    def do_request(i: int, seed_shift: int) -> dict:
        seed = seed_base + seed_shift + i
        started = time.perf_counter()
        try:
            m = time_one(base_url, headers, model, padder, context, max_tokens,
                         seed, timeout_s, state["thinking"])
            m["seed"] = seed
            m["error"] = ""
        except Exception as exc:  # noqa: BLE001 - per-request, recorded not raised
            m = {"error": str(exc) or exc.__class__.__name__, "seed": seed,
                 "prompt_tokens": -1, "completion_tokens": 0, "ttft_s": 0.0,
                 "prefill_tps": 0.0, "decode_tps": 0.0,
                 "e2e_s": 0.0, "e2e_tps": 0.0}
        m["start_abs"] = started
        m["end_abs"] = time.perf_counter()
        return m

    def run(indices, use_barrier: bool, seed_shift: int) -> list:
        n = len(indices)
        if use_barrier:
            barrier = threading.Barrier(n)

            def worker(i: int) -> dict:
                barrier.wait()
                return do_request(i, seed_shift)
        else:
            def worker(i: int) -> dict:  # noqa: E306
                return do_request(i, seed_shift)
        with ThreadPoolExecutor(max_workers=max(n, 1)) as pool:
            return list(pool.map(worker, indices))

    t0 = time.perf_counter()
    results = run(range(concurrency), use_barrier=True, seed_shift=0)

    had_404 = any("HTTP 404" in r["error"] for r in results if r["error"])
    had_400 = thinking_off and any("HTTP 400" in r["error"]
                                   for r in results if r["error"])
    if had_404:
        detail = next(r["error"] for r in results if "HTTP 404" in r["error"])
        raise RuntimeError(f"HTTP 404: {detail}")
    if had_400:
        state["thinking"] = False
        thinking_off = False
        time.sleep(1)  # let the engine settle before the retry pass
        failed_idx = [i for i, r in enumerate(results) if r["error"]]
        retries = run(failed_idx, use_barrier=False, seed_shift=RETRY_SEED_OFFSET)
        for idx, retry in zip(failed_idx, retries):
            results[idx] = retry

    def batch_wall() -> float:
        # if every request failed the retry pass still ran, so at least the
        # original end times anchor the measurement window
        ok = [r for r in results if not r["error"]]
        ref = ok or results
        return max(r["end_abs"] for r in ref) - t0

    return results, batch_wall(), thinking_off


def summarize(results: list, wall: float) -> dict:
    """Aggregate one batch: throughput + latency cells for the report."""
    ok = [r for r in results if not r["error"]]
    out = {
        "wall_s": wall,
        "requests": len(results),
        "failed": len(results) - len(ok),
    }
    if not ok:
        return out
    comp = sum(r["completion_tokens"] for r in ok)
    prom = sum(max(r["prompt_tokens"], 0) for r in ok)  # -1 = unavailable
    out["output_tps"] = comp / wall
    out["total_tps"] = (prom + comp) / wall
    out["req_s"] = len(ok) / wall
    out["ttft_p50_s"] = percentile([r["ttft_s"] for r in ok], 0.50)
    out["ttft_p95_s"] = percentile([r["ttft_s"] for r in ok], 0.95)
    out["e2e_p50_s"] = percentile([r["e2e_s"] for r in ok], 0.50)
    out["e2e_p95_s"] = percentile([r["e2e_s"] for r in ok], 0.95)
    out["decode_tps_mean"] = statistics.mean(r["decode_tps"] for r in ok)
    out["eff_concurrency"] = sum(r["end_abs"] - r["start_abs"] for r in ok) / wall
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--quick", action="store_true",
                    help="levels 1 and 8, 512-token context, 1 warm-up + "
                         "1 timed batch (sanity check)")
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
    contexts_raw = cfg("BENCH_CONTEXTS", ",".join(str(c) for c in DEFAULT_CONTEXTS))
    try:
        contexts = [int(x) for x in contexts_raw.split(",") if x.strip()]
    except ValueError as exc:
        sys.exit(f"invalid BENCH_CONTEXTS={contexts_raw!r} (want comma-separated "
                 f"integers): {exc}")
    runs = int(cfg("BENCH_RUNS", "3"))
    warmup = int(cfg("BENCH_WARMUP", "1"))
    max_tokens = int(cfg("BENCH_MAX_TOKENS", str(DEFAULT_MAX_TOKENS)))
    levels_raw = cfg("BENCH_LEVELS", ",".join(str(l) for l in DEFAULT_LEVELS))
    levels = sorted({int(x) for x in levels_raw.split(",") if x.strip()})
    if args.quick:
        levels = [1, 8]
        contexts = [512]
        runs = 1
        warmup = 1  # --quick is a fixed fast check, independent of BENCH_WARMUP
    if not levels or any(l < 1 for l in levels):
        sys.exit(f"no usable concurrency levels in BENCH_LEVELS={levels_raw!r}")
    if not contexts or any(c < 1 for c in contexts):
        sys.exit(f"no usable context sizes in BENCH_CONTEXTS={contexts_raw!r}")

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
    context_note = ("tokenizers not installed; padding sized by char heuristic"
                     if tokenizer_path is None else
                     "token-accurate padding via model tokenizer")

    # fit inside the served max model length (prompt + completion + template overhead)
    safety = 256
    ceiling = MAX_MODEL_LEN - max_tokens - safety
    contexts = sorted({min(c, ceiling) for c in contexts})

    # pool-capped plan: a level only runs at a context when the whole batch
    # (level x (context + max_tokens) tokens of KV) fits the pool
    plan = {c: [l for l in levels if l * (c + max_tokens) <= KV_POOL_TOKENS_APPROX]
            for c in contexts}
    if not any(plan.values()):
        sys.exit(f"no context x level combination fits the "
                 f"~{KV_POOL_TOKENS_APPROX // 1000}K-token KV pool — lower "
                 f"BENCH_CONTEXTS, BENCH_LEVELS, or BENCH_MAX_TOKENS")

    def fmt_ctx(c: int) -> str:
        return f"{c // 1024}K" if c >= 1024 and c % 1024 == 0 else str(c)

    transport = "httpx" if importlib.util.find_spec("httpx") else "urllib"

    print(f"base_url : {base_url} ({header_note})")
    print(f"model    : {model}")
    print(f"padding  : {context_note}")
    for ctx in contexts:
        skipped = [l for l in levels if l not in plan[ctx]]
        print(f"ctx {fmt_ctx(ctx):>6} : levels {plan[ctx]}"
              + (f"  (skip {skipped}: beyond the "
                 f"~{KV_POOL_TOKENS_APPROX // 1000}K KV pool)" if skipped else ""))
    print(f"max_tokens={max_tokens}, {warmup} warm-up + {runs} timed batch(es) "
          f"per context x level combo")

    gpu_before = gpu_snapshot()
    if gpu_before:
        print(f"gpu      : {gpu_before['name']} | {gpu_before['util_pct']}% | "
              f"{gpu_before['mem_used']} | {gpu_before['power_w']} W | "
              f"{gpu_before['temp_c']} C")

    def failed_summary(s: dict) -> bool:
        return s["failed"] == s["requests"] and s["requests"] > 0

    seed_base = int(time.time()) % 100000
    thinking_off = True  # once a 400 retry happens, the flag stays off
    combos: dict[int, dict[int, list]] = {
        ctx: {lvl: [] for lvl in plan[ctx]} for ctx in contexts
    }
    all_requests = []
    stop_all = False
    try:
        for ci, ctx in enumerate(contexts):
            for li, lvl in enumerate(plan[ctx]):
                if stop_all:
                    break
                tag = f"[{fmt_ctx(ctx)} c={lvl:>2}]"
                for b in range(warmup + runs):
                    batch_seed = seed_base + ci * 10_000_000 + li * 100_000 + b * 1000
                    role = "warmup" if b < warmup else "timed"
                    print(f"{tag} {role} batch {b + 1}/{warmup + runs} ...", flush=True)
                    try:
                        results, wall, thinking_off = fire_batch(
                            base_url, headers, model, padder, ctx, max_tokens,
                            batch_seed, timeout_s, thinking_off, lvl)
                    except RuntimeError as exc:
                        # fire_batch only raises on HTTP 404 (unknown model)
                        sys.exit(f"{tag} HTTP 404 from {base_url} — check BENCH_MODEL "
                                 f"against the served model name. Detail: {exc}")
                    for r in results:
                        r["level"] = lvl
                        r["context"] = ctx
                        r["batch"] = b
                        r["role"] = role
                    all_requests.extend(results)
                    n_fail = sum(1 for r in results if r["error"])
                    if role == "timed":
                        s = summarize(results, wall)
                        combos[ctx][lvl].append(s)
                        print(f"{tag} wall={wall:.2f}s out={s.get('output_tps', 0):8.1f} "
                              f"tok/s total={s.get('total_tps', 0):8.1f} tok/s "
                              f"TTFT p50={s.get('ttft_p50_s', 0):.2f}s "
                              f"E2E p95={s.get('e2e_p95_s', 0):.2f}s"
                              + (f"  [{n_fail} failed]" if n_fail else ""), flush=True)
                        # stop early when the endpoint rejects every request in a
                        # timed batch (same as context_bench's all-runs-failed break)
                        if n_fail == len(results):
                            print(f"{tag} all {lvl} requests failed — "
                                  f"{results[0]['error']} "
                                  f"(stopping the suite)", flush=True)
                            stop_all = True
                            break
                    elif n_fail:
                        print(f"{tag} warm-up: {n_fail}/{len(results)} request(s) "
                              f"failed (continuing)", flush=True)
                if stop_all:
                    break
    except KeyboardInterrupt:
        print("\ninterrupted — writing partial results", flush=True)

    gpu_after = gpu_snapshot()

    requests_json = [
        {k: r[k] for k in ("context", "level", "batch", "role", "seed",
                           "prompt_tokens", "completion_tokens", "ttft_s",
                           "prefill_tps", "decode_tps", "e2e_s", "e2e_tps",
                           "error")}
        for r in all_requests
    ]
    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "endpoint": base_url,
        "auth": "basic" if basic else "bearer",
        "model": model,
        "transport": transport,
        "concurrency_levels": levels,
        "context_targets": contexts,
        "kv_pool_tokens_approx": KV_POOL_TOKENS_APPROX,
        "max_tokens": max_tokens,
        "runs_per_combo": runs,
        "warmup_per_combo": warmup,
        "notes": context_note,
        "gpu_before": gpu_before,
        "gpu_after": gpu_after,
        "contexts": [
            {
                "context": ctx,
                "levels": [
                    {"concurrency": lvl, "batches": combos[ctx][lvl]}
                    for lvl in plan[ctx]
                ],
            }
            for ctx in contexts
        ],
        "requests": requests_json,
    }
    out_dir = REPO_ROOT / "benchmarks" / "results"
    out_dir.mkdir(exist_ok=True)
    stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    for name in (f"{stamp}_parallel.json", "latest_parallel.json"):
        (out_dir / name).write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nresults: {out_dir / (stamp + '_parallel.json')}")

    # markdown: one table per context; throughput/wall cells are medians
    # over the timed batches of the combo, latency cells pool every timed
    # request of it
    print("\n## Markdown (paste into README)\n")
    for ctx in contexts:
        if not plan[ctx]:
            continue
        print(f"### {fmt_ctx(ctx)} context\n")
        print("| concurrency | wall (s) | req/s | output (tok/s) | total (tok/s) "
              "| TTFT p50 (s) | TTFT p95 (s) | E2E p50 (s) | E2E p95 (s) | decode (tok/s/req) |")
        print("|---|---|---|---|---|---|---|---|---|---|")
        for lvl in plan[ctx]:
            good = [s for s in combos[ctx][lvl] if not failed_summary(s)]
            if not good:
                print(f"| {lvl} | — | — | — | — | — | — | — | — | — |")
                continue
            reqs = [r for r in all_requests
                   if r["context"] == ctx and r["level"] == lvl
                   and r["role"] == "timed" and not r["error"]]
            pool_ttft = [r["ttft_s"] for r in reqs]
            pool_e2e = [r["e2e_s"] for r in reqs]
            decode_mean = statistics.mean(r["decode_tps"] for r in reqs) \
                if reqs else 0.0
            print(f"| {lvl} | {statistics.median([x['wall_s'] for x in good]):.2f} | "
                  f"{statistics.median([x['req_s'] for x in good]):.2f} | "
                  f"{statistics.median([x['output_tps'] for x in good]):,.0f} | "
                  f"{statistics.median([x['total_tps'] for x in good]):,.0f} | "
                  f"{percentile(pool_ttft, 0.50):.2f} | {percentile(pool_ttft, 0.95):.2f} | "
                  f"{percentile(pool_e2e, 0.50):.2f} | {percentile(pool_e2e, 0.95):.2f} | "
                  f"{decode_mean:.1f} |")
        print()
    n_fail = sum(1 for r in all_requests if r["error"])
    print(f"\nmedians over the timed batches ({runs} per context x level combo; "
          f"all requests of a batch fire simultaneously, unique padding per "
          f"request: cold prefill despite --enable-prefix-caching; levels are "
          f"pool-capped per context so no batch exceeds the "
          f"~{KV_POOL_TOKENS_APPROX // 1000}K-token KV pool)"
          + (f"; {n_fail} failed request(s) excluded and recorded in the JSON"
             if n_fail else ""))
    if all_requests and all(r["error"] for r in all_requests):
        sys.exit("all requests failed — no measurements to report")
    return 0


if __name__ == "__main__":
    sys.exit(main())
