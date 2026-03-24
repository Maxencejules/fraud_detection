import os
import time
import asyncio
import pandas as pd
import numpy as np
import mlflow.pyfunc
import redis.asyncio as aioredis
from typing import Optional, List
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
import uvicorn

# Constants
MODEL_URI = os.getenv("MODEL_URI", "models:/fraud-detector/Production")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
THRESHOLD_REVIEW = float(os.getenv("THRESHOLD_REVIEW", "0.4"))
THRESHOLD_BLOCK = float(os.getenv("THRESHOLD_BLOCK", "0.75"))
WORKERS = int(os.getenv("WORKERS", "1"))

FEATURE_COLS = [
    "amount", "amount_log", "amount_zscore", "tx_count_1h", "tx_count_24h",
    "tx_sum_1h", "tx_sum_24h", "unique_merchants_24h", "unique_countries_7d",
    "hour_of_day", "day_of_week", "is_weekend", "card_present",
    "merchant_fraud_rate_30d", "user_chargeback_rate"
]

class AppState:
    def __init__(self):
        self.model = None
        self.model_version: str = "unknown"
        self.redis: Optional[aioredis.Redis] = None

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
    transaction_id: str
    fraud_probability: float
    decision: str
    model_version: str
    latency_ms: float

def load_champion_model():
    print(f"Loading model from {MODEL_URI}...")
    try:
        model = mlflow.pyfunc.load_model(MODEL_URI)
        # Attempt to extract version from model metadata if available
        version = "Production"
        if hasattr(model, "metadata") and hasattr(model.metadata, "version"):
            version = model.metadata.version
        return model, str(version)
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, "error"

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    state.redis = aioredis.from_url(REDIS_URL, decode_responses=True)
    state.model, state.model_version = await asyncio.to_thread(load_champion_model)
    yield
    # Shutdown
    await state.redis.close()

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
    elif prob < THRESHOLD_BLOCK:
        return "REVIEW"
    else:
        return "BLOCK"

@app.get("/health")
async def health():
    return {
        "model_loaded": state.model is not None,
        "model_version": state.model_version,
        "status": "healthy"
    }

@app.post("/reload-model")
async def reload_model():
    state.model, state.model_version = await asyncio.to_thread(load_champion_model)
    if state.model is None:
        raise HTTPException(status_code=500, detail="Failed to reload model")
    return {"status": "success", "model_version": state.model_version}

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: InferenceRequest):
    start_time = time.perf_counter()
    
    if state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Prepare DataFrame
    data = [[getattr(request, col) for col in FEATURE_COLS]]
    df = pd.DataFrame(data, columns=FEATURE_COLS)

    # Run inference in thread pool
    try:
        raw_pred = await asyncio.to_thread(state.model.predict, df)
        
        # Handle shapes (n,) and (n, 2)
        if isinstance(raw_pred, np.ndarray):
            if raw_pred.ndim == 2 and raw_pred.shape[1] == 2:
                prob = float(raw_pred[0][1])  # Prob of class 1
            else:
                prob = float(raw_pred[0])
        else:
            # Assume pandas Series or list
            prob = float(raw_pred[0])
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

    decision = get_decision(prob)
    latency_ms = (time.perf_counter() - start_time) * 1000

    # Cache result
    try:
        cache_val = f"{prob}:{decision}"
        await state.redis.setex(f"pred:{request.transaction_id}", 3600, cache_val)
    except Exception as e:
        print(f"Redis cache error: {e}")

    return PredictionResponse(
        transaction_id=request.transaction_id,
        fraud_probability=round(prob, 4),
        decision=decision,
        model_version=state.model_version,
        latency_ms=round(latency_ms, 2)
    )

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, workers=WORKERS)
