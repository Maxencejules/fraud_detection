import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any, Optional

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

try:
    import mlflow
except ImportError:  # pragma: no cover - exercised in lightweight test envs
    mlflow = None

try:
    import redis.asyncio as aioredis
except ImportError:  # pragma: no cover - exercised in lightweight test envs
    aioredis = None

from shared.config import FEATURE_COLS
from shared.observability import configure_logging, log_event

# Constants
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
MODEL_URI = os.getenv("MODEL_URI", "models:/fraud-detector/Production")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
THRESHOLD_REVIEW = float(os.getenv("THRESHOLD_REVIEW", "0.4"))
THRESHOLD_BLOCK = float(os.getenv("THRESHOLD_BLOCK", "0.75"))
WORKERS = int(os.getenv("WORKERS", "1"))
logger = configure_logging("predictor")

PREDICT_REQUESTS = Counter(
    "fraud_predict_requests_total",
    "Prediction requests served by the predictor.",
    ["decision"],
)
PREDICT_ERRORS = Counter(
    "fraud_predict_errors_total",
    "Prediction request failures grouped by error type.",
    ["error_type"],
)
PREDICT_LATENCY = Histogram(
    "fraud_predict_latency_seconds",
    "Prediction latency for the FastAPI predictor.",
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 1.0, 5.0),
)
MODEL_RELOADS = Counter(
    "fraud_predict_model_reloads_total",
    "Number of attempted model reloads.",
    ["status"],
)
REDIS_CACHE_ERRORS = Counter(
    "fraud_predict_redis_cache_errors_total",
    "Number of Redis cache write failures in the predictor.",
)
MODEL_LOADED = Gauge(
    "fraud_predict_model_loaded",
    "Whether a serving model is currently loaded.",
)


class AppState:
    def __init__(self):
        self.model: Any = None
        self.model_version: str = "unknown"
        self.redis: Optional[Any] = None


state = AppState()


class InferenceRequest(BaseModel):
    transaction_id: str
    amount: float
    amount_log: float
    amount_zscore: float
    tx_count_1h: int
    tx_count_24h: int
    tx_sum_1h: float
    tx_sum_24h: float
    unique_merchants_24h: int
    unique_countries_7d: int
    hour_of_day: int
    day_of_week: int
    is_weekend: bool
    card_present: bool
    merchant_fraud_rate_30d: float
    user_chargeback_rate: float


class PredictionResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    transaction_id: str
    fraud_probability: float
    decision: str
    model_version: str
    latency_ms: float


def load_champion_model() -> tuple[Any, str]:
    if mlflow is None:
        log_event(logger, logging.WARNING, "model_load_skipped", reason="mlflow_not_installed")
        MODEL_LOADED.set(0)
        return None, "unavailable"

    if MLFLOW_TRACKING_URI:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    log_event(logger, logging.INFO, "model_load_started", model_uri=MODEL_URI)
    try:
        loaded_model = mlflow.pyfunc.load_model(MODEL_URI)
        version = getattr(getattr(loaded_model, "metadata", None), "run_id", None) or MODEL_URI
        MODEL_LOADED.set(1)
        log_event(logger, logging.INFO, "model_load_succeeded", model_version=str(version))
        return loaded_model, str(version)
    except Exception as exc:
        MODEL_LOADED.set(0)
        log_event(logger, logging.ERROR, "model_load_failed", error=str(exc), model_uri=MODEL_URI)
        return None, "error"


def extract_probability(raw_pred: Any) -> float:
    if hasattr(raw_pred, "to_numpy"):
        raw_pred = raw_pred.to_numpy()

    arr = np.asarray(raw_pred)
    if arr.size == 0:
        raise ValueError("Model returned an empty prediction array")

    if arr.ndim == 0:
        prob = float(arr.item())
    elif arr.ndim == 1:
        if arr.size == 2 and np.all((arr >= 0.0) & (arr <= 1.0)):
            prob = float(arr[-1])
        else:
            prob = float(arr[0])
    else:
        first_row = np.asarray(arr[0])
        if first_row.size == 0:
            raise ValueError("Model returned an empty prediction row")
        prob = float(first_row[-1] if first_row.size > 1 else first_row[0])

    return float(np.clip(prob, 0.0, 1.0))


@asynccontextmanager
async def lifespan(app: FastAPI):
    if state.redis is None and aioredis is not None:
        state.redis = aioredis.from_url(REDIS_URL, decode_responses=True)
        log_event(logger, logging.INFO, "redis_client_initialized", redis_url=REDIS_URL)

    if state.model is None and state.model_version == "unknown":
        state.model, state.model_version = await asyncio.to_thread(load_champion_model)

    yield

    redis_client = state.redis
    if redis_client is not None:
        if hasattr(redis_client, "aclose"):
            await redis_client.aclose()
        elif hasattr(redis_client, "close"):
            maybe_coro = redis_client.close()
            if asyncio.iscoroutine(maybe_coro):
                await maybe_coro


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_decision(prob: float) -> str:
    if prob < THRESHOLD_REVIEW:
        return "APPROVE"
    if prob < THRESHOLD_BLOCK:
        return "REVIEW"
    return "BLOCK"


@app.get("/health")
async def health():
    return {
        "model_loaded": state.model is not None,
        "model_version": state.model_version,
        "status": "healthy",
    }


@app.get("/ready")
async def ready():
    if state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return {"status": "ready", "model_version": state.model_version}


@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/reload-model")
async def reload_model():
    state.model, state.model_version = await asyncio.to_thread(load_champion_model)
    if state.model is None:
        MODEL_RELOADS.labels(status="failed").inc()
        raise HTTPException(status_code=500, detail="Failed to reload model")
    MODEL_RELOADS.labels(status="success").inc()
    return {"status": "success", "model_version": state.model_version}


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: InferenceRequest, response: Response):
    start_time = time.perf_counter()

    if state.model is None:
        PREDICT_ERRORS.labels(error_type="model_unavailable").inc()
        raise HTTPException(status_code=503, detail="Model not loaded")

    df = pd.DataFrame([[getattr(request, col) for col in FEATURE_COLS]], columns=FEATURE_COLS)

    try:
        raw_pred = await asyncio.to_thread(state.model.predict, df)
        prob = extract_probability(raw_pred)
    except Exception as exc:
        PREDICT_ERRORS.labels(error_type="inference").inc()
        log_event(logger, logging.ERROR, "prediction_failed", error=str(exc), transaction_id=request.transaction_id)
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}") from exc

    decision = get_decision(prob)
    latency_ms = (time.perf_counter() - start_time) * 1000
    rounded_latency = round(latency_ms, 2)
    PREDICT_LATENCY.observe(latency_ms / 1000.0)
    PREDICT_REQUESTS.labels(decision=decision).inc()
    response.headers["X-Latency-Ms"] = f"{rounded_latency:.2f}"

    if state.redis is not None:
        try:
            await state.redis.setex(f"pred:{request.transaction_id}", 3600, f"{prob}:{decision}")
        except Exception as exc:
            REDIS_CACHE_ERRORS.inc()
            log_event(
                logger,
                logging.ERROR,
                "redis_cache_error",
                error=str(exc),
                transaction_id=request.transaction_id,
            )

    log_event(
        logger,
        logging.INFO,
        "prediction_served",
        decision=decision,
        latency_ms=rounded_latency,
        model_version=state.model_version,
        transaction_id=request.transaction_id,
    )

    return PredictionResponse(
        transaction_id=request.transaction_id,
        fraud_probability=round(prob, 4),
        decision=decision,
        model_version=state.model_version,
        latency_ms=rounded_latency,
    )


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, workers=WORKERS)
