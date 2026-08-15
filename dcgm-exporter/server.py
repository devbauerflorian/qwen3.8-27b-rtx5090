#!/usr/bin/env python3
"""Minimal NVML -> Prometheus exporter for per-GPU metrics.

Drop-in replacement for the official NVIDIA DCGM exporter for stacks
without nvcr.io (NGC registry) access. It exports the same DCGM_FI_DEV_*
metric names on :9400, so the prometheus "dcgm" job, the caddy
/dcgm/metrics route, and the Grafana dashboard all work unchanged.

The NVML driver library is injected into the container by the NVIDIA
container toolkit (the service reserves the GPU in docker-compose.yml);
nothing NVIDIA needs to be installed in this image.

Serves GET /metrics on 0.0.0.0:$DCGM_EXPORTER_PORT (default 9400).
"""

import http.server
import os
import time

import pynvml

PORT = int(os.environ.get("DCGM_EXPORTER_PORT", "9400"))

# Metric name -> (HELP text, TYPE)
METRICS = {
    "DCGM_FI_DEV_GPU_UTIL": ("GPU utilization (percent)", "gauge"),
    "DCGM_FI_DEV_MEMORY_USED": ("GPU memory used (MiB)", "gauge"),
    "DCGM_FI_DEV_MEMORY_TOTAL": ("GPU memory total (MiB)", "gauge"),
    "DCGM_FI_DEV_POWER_USAGE": ("GPU power usage (W)", "gauge"),
    "DCGM_FI_DEV_GPU_TEMP": ("GPU temperature (C)", "gauge"),
    "DCGM_FI_DEV_FAN_SPEED": ("GPU fan speed (percent)", "gauge"),
    "DCGM_FI_DEV_CLOCK_SM": ("GPU SM clock (MHz)", "gauge"),
}

MIB = 2.0 ** 20


def read_gpu(device):
    """Read one GPU; a failed individual metric is dropped, not fatal."""
    handle = pynvml.nvmlDeviceGetHandleByIndex(device)
    values = {}

    def try_set(name, fn):
        try:
            values[name] = fn()
        except pynvml.NVMLError:
            pass

    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
    memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
    values["DCGM_FI_DEV_GPU_UTIL"] = float(utilization.gpu)
    values["DCGM_FI_DEV_MEMORY_USED"] = memory.used / MIB
    values["DCGM_FI_DEV_MEMORY_TOTAL"] = memory.total / MIB
    try_set("DCGM_FI_DEV_POWER_USAGE", lambda: pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0)
    try_set("DCGM_FI_DEV_GPU_TEMP", lambda: pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU))
    try_set("DCGM_FI_DEV_FAN_SPEED", lambda: pynvml.nvmlDeviceGetFanSpeed(handle))
    try_set("DCGM_FI_DEV_CLOCK_SM", lambda: pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM))

    label = str(pynvml.nvmlDeviceGetUUID(handle))
    return label, values


def render():
    lines = []
    for name, (help_text, metric_type) in METRICS.items():
        lines.append(f"# HELP {name} {help_text}")
        lines.append(f"# TYPE {name} {metric_type}")

    try:
        devices = [read_gpu(i) for i in range(pynvml.nvmlDeviceGetCount())]
    except pynvml.NVMLError as exc:
        print(f"nvml read failed: {exc}", flush=True)
        return "\n".join(lines) + "\n"

    if not devices:
        print("no GPU visible", flush=True)

    for name in METRICS:
        for label, values in devices:
            if name in values:
                lines.append(f'{name}{{gpu="{label}"}} {values[name]}')
    return "\n".join(lines) + "\n"


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path in ("/metrics", "/"):
            body = (render() if self.path == "/metrics" else "ok\n").encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, *args):
        pass


def main():
    while True:
        try:
            pynvml.nvmlInit()
            break
        except pynvml.NVMLError as exc:
            print(f"nvml not ready yet ({exc}); retrying in 10s", flush=True)
            time.sleep(10)

    print(f"listening on 0.0.0.0:{PORT}", flush=True)
    http.server.ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
