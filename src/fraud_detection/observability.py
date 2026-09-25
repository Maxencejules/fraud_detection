"""Structured JSON logging and Prometheus helpers shared by all services."""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any

from prometheus_client import start_http_server

_NOISY_LOGGERS = ("urllib3", "httpx", "httpcore", "mlflow", "alembic", "git", "numba")
# Expected in slim containers (no git binary / no pip in uv-built environments).
_SILENCED_LOGGERS = ("mlflow.utils.git_utils", "mlflow.utils.environment")
_RESERVED = frozenset(logging.makeLogRecord({}).__dict__) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    """One JSON object per line; extra fields passed via ``extra=`` are included."""

    def __init__(self, service: str) -> None:
        super().__init__()
        self.service = service

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "service": self.service,
            "logger": record.name,
            "event": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(service: str, level: str = "INFO") -> logging.Logger:
    """Route all logging (including third-party libraries) through one JSON handler."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter(service))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level.upper())
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(max(logging.WARNING, root.level))
    for name in _SILENCED_LOGGERS:
        logging.getLogger(name).setLevel(logging.ERROR)
    return logging.getLogger(f"fraud_detection.{service}")


def start_metrics_server(port: int, logger: logging.Logger) -> None:
    """Expose ``/metrics`` for Prometheus on ``port`` (disabled when ``port`` is 0)."""
    if port <= 0:
        return
    start_http_server(port)
    logger.info("metrics_server_started", extra={"port": port})
