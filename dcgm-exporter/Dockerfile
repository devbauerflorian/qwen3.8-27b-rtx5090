# Minimal NVML -> Prometheus exporter for per-GPU metrics.
#
# Drop-in replacement for the official nvcr.io/nvidia/dcgm-exporter image:
# it exports the same DCGM_FI_DEV_* metric names on :9400, so the
# prometheus "dcgm" job and the caddy /dcgm/metrics route are unchanged.
# Public base image + pypi dep only — no nvcr.io (NGC) login required.
#
# The NVML driver library (libnvidia-ml.so) is injected into the container
# by the NVIDIA container toolkit because the service reserves the GPU
# (deploy.resources.reservations.devices in docker-compose.yml).
# Build context: repo subdirectory dcgm-exporter/ (see docker-compose.yml).

FROM python:3.13-slim

# Pinned like the rest of the stack — bump deliberately (see README "Pinned versions").
RUN pip install --no-cache-dir nvidia-ml-py==13.610.43

COPY server.py /server.py

EXPOSE 9400
CMD ["python3", "/server.py"]
