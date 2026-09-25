# syntax=docker/dockerfile:1.7
#
# One image definition for every Python service. Each service installs only the
# dependency group it needs from the locked environment (uv.lock):
#
#   EXTRAS=stream      producer, feature-processor, scorer
#   EXTRAS=serving     predictor                 (NEEDS_OPENMP=true)
#   EXTRAS=training    bootstrap, trainer        (NEEDS_OPENMP=true)
#   EXTRAS=monitoring  monitor
#
# The runtime stage contains no compilers or build tools, and runs as an unprivileged user.

ARG PYTHON_IMAGE=python:3.12-slim-bookworm

FROM ghcr.io/astral-sh/uv:0.12.19 AS uv

# --- builder: resolve the locked dependencies into /opt/venv -------------------------
FROM ${PYTHON_IMAGE} AS builder
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv
ARG EXTRAS=stream
WORKDIR /src

# Dependencies first, so source edits do not invalidate this layer.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project $(for e in ${EXTRAS}; do printf -- '--extra %s ' "$e"; done)

COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable $(for e in ${EXTRAS}; do printf -- '--extra %s ' "$e"; done)

# --- runtime --------------------------------------------------------------------------
FROM ${PYTHON_IMAGE} AS runtime
ARG NEEDS_OPENMP=false
# LightGBM links against the system OpenMP runtime.
RUN if [ "${NEEDS_OPENMP}" = "true" ]; then \
        apt-get update \
        && apt-get install -y --no-install-recommends libgomp1 \
        && rm -rf /var/lib/apt/lists/*; \
    fi \
    && groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --home-dir /app --shell /usr/sbin/nologin app \
    && mkdir -p /app/data /app/reports \
    && chown -R app:app /app

ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

COPY --from=builder /opt/venv /opt/venv
WORKDIR /app
USER app

# The service to run is chosen by the compose file (e.g. `python -m fraud_detection.scorer`).
CMD ["python", "-c", "import fraud_detection; print(fraud_detection.__version__)"]
