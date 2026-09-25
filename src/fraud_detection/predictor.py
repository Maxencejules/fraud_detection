"""Online fraud scoring API.

* Serves the model behind the registry alias (``models:/<name>@champion``) and polls
  the registry so a newly promoted version is picked up without a restart. A candidate
  model is only swapped in after its feature contract has been checked.
* Stateless: no per-request writes, so replicas scale horizontally.
* Inference runs on one dedicated thread. Model calls never block the event loop and
  never run concurrently on the same booster.
"""

# No ``from __future__ import annotations``: FastAPI must resolve the ``Depends`` in
# closure-scoped annotations at definition time.
import asyncio
import logging
import secrets
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager, suppress
from typing import Annotated, Any

import numpy as np
from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, Info, generate_latest

from fraud_detection import __version__
from fraud_detection.config import DecisionThresholds, PredictorSettings
from fraud_detection.features import FEATURE_COLUMNS
from fraud_detection.observability import configure_logging
from fraud_detection.registry import LoadedModel, ModelRef
from fraud_detection.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    PredictionRequest,
    PredictionResponse,
    ScoredTransaction,
)

logger = logging.getLogger("fraud_detection.predictor")

LATENCY_BUCKETS = (0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0)
PREDICTIONS = Counter(
    "fraud_predictor_predictions_total", "Transactions scored, by decision.", ["decision"]
)
REQUEST_LATENCY = Histogram(
    "fraud_predictor_request_latency_seconds",
    "Handler latency, by endpoint.",
    ["endpoint"],
    buckets=LATENCY_BUCKETS,
)
INFERENCE_LATENCY = Histogram(
    "fraud_predictor_inference_latency_seconds",
    "Model inference time per request (queueing included).",
    buckets=LATENCY_BUCKETS,
)
BATCH_SIZE = Histogram(
    "fraud_predictor_batch_size",
    "Items per batch request.",
    buckets=(1, 5, 10, 25, 50, 100, 250, 500, 1_000),
)
ERRORS = Counter("fraud_predictor_errors_total", "Failed requests, by type.", ["type"])
MODEL_RELOADS = Counter(
    "fraud_predictor_model_reloads_total", "Model load attempts, by outcome.", ["outcome"]
)
MODEL_LOADED = Gauge("fraud_predictor_model_loaded", "1 when a model is being served.")
MODEL_INFO = Info("fraud_predictor_model", "Currently served model.")

Resolver = Callable[[], ModelRef | None]
Loader = Callable[[str, str | None], LoadedModel]


class ModelManager:
    """Owns the serving model: initial load, alias polling and atomic hot-swaps."""

    def __init__(
        self,
        *,
        resolver: Resolver,
        loader: Loader,
        poll_interval_s: float,
        expected_features: Sequence[str] = FEATURE_COLUMNS,
    ) -> None:
        self._resolver = resolver
        self._loader = loader
        self._poll_interval_s = poll_interval_s
        self._expected_features = tuple(expected_features)
        self._lock = asyncio.Lock()
        self._task: asyncio.Task[None] | None = None
        self.current: LoadedModel | None = None
        self.last_error: str | None = None

    async def start(self) -> None:
        await self.refresh()
        if self._poll_interval_s > 0:
            self._task = asyncio.create_task(self._poll(), name="model-poller")

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with suppress(asyncio.CancelledError):
                await self._task

    async def refresh(self, force: bool = False) -> bool:
        """Load the model the registry points to, if it changed. Returns True on swap."""
        async with self._lock:
            try:
                ref = await asyncio.to_thread(self._resolver)
                if ref is None:
                    self.last_error = "no model registered under the configured alias"
                    return False
                if not force and self.current is not None and self.current.uri == ref.uri:
                    return False
                candidate = await asyncio.to_thread(self._loader, ref.uri, ref.version)
                if candidate.feature_columns != self._expected_features:
                    raise ValueError(
                        f"model features {candidate.feature_columns} do not match the "
                        f"service contract {self._expected_features}"
                    )
            except Exception as exc:
                self.last_error = f"{type(exc).__name__}: {exc}"
                MODEL_RELOADS.labels(outcome="failed").inc()
                logger.exception("model_load_failed")
                return False
            self.current = candidate
            self.last_error = None
            MODEL_RELOADS.labels(outcome="success").inc()
            MODEL_LOADED.set(1)
            MODEL_INFO.info({"version": candidate.version, "run_id": candidate.run_id or ""})
            logger.info("model_loaded", extra={"version": candidate.version, "uri": candidate.uri})
            return True

    async def _poll(self) -> None:
        while True:
            await asyncio.sleep(self._poll_interval_s)
            await self.refresh()


def registry_manager(settings: PredictorSettings) -> ModelManager:
    """Production wiring: resolve through the MLflow registry."""
    import mlflow
    from mlflow.tracking import MlflowClient

    from fraud_detection.registry import load_model, resolve_alias

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    fixed_uri = settings.model_uri

    def resolve() -> ModelRef | None:
        if fixed_uri:
            return ModelRef(name=settings.model_name, version=fixed_uri, run_id=None, uri=fixed_uri)
        return resolve_alias(MlflowClient(), settings.model_name, settings.model_alias)

    return ModelManager(
        resolver=resolve,
        loader=load_model,
        poll_interval_s=0.0 if fixed_uri else settings.model_poll_interval_s,
    )


def create_app(settings: PredictorSettings, manager: ModelManager | None = None) -> FastAPI:
    model_manager = manager or registry_manager(settings)
    thresholds: DecisionThresholds = settings.thresholds()
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="inference")

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        await model_manager.start()
        yield
        await model_manager.stop()
        executor.shutdown(wait=False, cancel_futures=True)

    app = FastAPI(
        title="Fraud Detection Predictor",
        version=__version__,
        description="Scores card transactions with the champion fraud model.",
        lifespan=lifespan,
    )
    if settings.cors_allow_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_allow_origins,
            allow_methods=["GET", "POST"],
            allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
        )

    @app.middleware("http")
    async def request_context(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """Propagate ``X-Request-ID`` and turn unexpected errors into opaque 500s."""
        rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex
        try:
            response = await call_next(request)
        except Exception:
            ERRORS.labels(type="internal").inc()
            logger.exception("unhandled_error", extra={"request_id": rid, "path": request.url.path})
            response = JSONResponse(
                status_code=500, content={"detail": "Internal server error", "request_id": rid}
            )
        response.headers["X-Request-ID"] = rid
        return response

    def serving_model() -> LoadedModel:
        model = model_manager.current
        if model is None:
            ERRORS.labels(type="model_unavailable").inc()
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Model not loaded")
        return model

    async def score(model: LoadedModel, items: Sequence[PredictionRequest]) -> np.ndarray:
        matrix = np.array(
            [[float(getattr(item, column)) for column in FEATURE_COLUMNS] for item in items],
            dtype=np.float64,
        )
        started = time.perf_counter()
        loop = asyncio.get_running_loop()
        probabilities = await loop.run_in_executor(executor, model.scorer.predict_proba, matrix)
        INFERENCE_LATENCY.observe(time.perf_counter() - started)
        return probabilities

    def scored(item: PredictionRequest, probability: float) -> ScoredTransaction:
        decision = thresholds.decide(probability)
        PREDICTIONS.labels(decision=decision.value).inc()
        return ScoredTransaction(
            transaction_id=item.transaction_id,
            fraud_probability=round(probability, 6),
            decision=decision,
        )

    @app.get("/health", tags=["operations"])
    async def health() -> dict[str, str]:
        """Liveness: the process is up (a model may not be loaded yet)."""
        return {"status": "ok"}

    @app.get("/ready", tags=["operations"])
    async def ready() -> JSONResponse:
        """Readiness: a model is loaded and requests can be served."""
        model = model_manager.current
        if model is None:
            return JSONResponse(
                status_code=503,
                content={"status": "not_ready", "reason": model_manager.last_error},
            )
        return JSONResponse(content={"status": "ready", "model_version": model.version})

    @app.get("/metrics", tags=["operations"])
    async def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.get("/v1/model", tags=["model"])
    async def model_info(model: Annotated[LoadedModel, Depends(serving_model)]) -> dict[str, Any]:
        return {
            "name": settings.model_name,
            "alias": settings.model_alias,
            "version": model.version,
            "uri": model.uri,
            "run_id": model.run_id,
            "loaded_at": model.loaded_at,
            "feature_columns": list(model.feature_columns),
            "thresholds": thresholds.model_dump(),
        }

    @app.post("/v1/predict", response_model=PredictionResponse, tags=["scoring"])
    async def predict(
        payload: PredictionRequest,
        response: Response,
        model: Annotated[LoadedModel, Depends(serving_model)],
    ) -> PredictionResponse:
        started = time.perf_counter()
        probability = float((await score(model, [payload]))[0])
        result = scored(payload, probability)
        latency_ms = (time.perf_counter() - started) * 1000
        REQUEST_LATENCY.labels(endpoint="predict").observe(latency_ms / 1000)
        response.headers["X-Latency-Ms"] = f"{latency_ms:.3f}"
        return PredictionResponse(
            **result.model_dump(), model_version=model.version, latency_ms=round(latency_ms, 3)
        )

    @app.post("/v1/predict/batch", response_model=BatchPredictionResponse, tags=["scoring"])
    async def predict_batch(
        payload: BatchPredictionRequest,
        model: Annotated[LoadedModel, Depends(serving_model)],
    ) -> BatchPredictionResponse:
        if len(payload.items) > settings.max_batch_size:
            ERRORS.labels(type="batch_too_large").inc()
            raise HTTPException(
                status.HTTP_413_CONTENT_TOO_LARGE,
                f"batch of {len(payload.items)} exceeds the limit of {settings.max_batch_size}",
            )
        started = time.perf_counter()
        BATCH_SIZE.observe(len(payload.items))
        probabilities = await score(model, payload.items)
        items = [scored(i, float(p)) for i, p in zip(payload.items, probabilities, strict=True)]
        latency_ms = (time.perf_counter() - started) * 1000
        REQUEST_LATENCY.labels(endpoint="predict_batch").observe(latency_ms / 1000)
        return BatchPredictionResponse(
            items=items, model_version=model.version, latency_ms=round(latency_ms, 3)
        )

    @app.post("/v1/admin/reload", tags=["operations"])
    async def reload_model(
        authorization: Annotated[str | None, Header()] = None,
    ) -> dict[str, Any]:
        """Force a registry lookup and reload. Requires ``Authorization: Bearer <ADMIN_TOKEN>``."""
        if settings.admin_token is None:
            raise HTTPException(status.HTTP_403_FORBIDDEN, "Admin API disabled (ADMIN_TOKEN unset)")
        expected = f"Bearer {settings.admin_token.get_secret_value()}"
        if authorization is None or not secrets.compare_digest(authorization, expected):
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid admin token")
        swapped = await model_manager.refresh(force=True)
        model = model_manager.current
        if model is None:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, model_manager.last_error)
        return {"reloaded": swapped, "model_version": model.version}

    return app


def main() -> None:
    import uvicorn

    settings = PredictorSettings()
    configure_logging("predictor", settings.log_level)
    uvicorn.run(
        create_app(settings),
        host=settings.host,
        port=settings.port,
        log_config=None,
        access_log=settings.access_log,
        server_header=False,
        timeout_graceful_shutdown=10,
    )


if __name__ == "__main__":
    main()
