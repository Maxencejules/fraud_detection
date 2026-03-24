import time
from typing import Optional
from pydantic import BaseModel, Field

class RawTransaction(BaseModel):
    transaction_id: str
    user_id: str
    merchant_id: str
    amount: float
    currency: str = "USD"
    timestamp: float = Field(default_factory=time.time)
    merchant_category: str
    country: str
    card_present: bool
    device_fingerprint: Optional[str] = None

class FeatureVector(BaseModel):
    transaction_id: str
    user_id: str
    timestamp: float
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
    merchant_category: str
    merchant_fraud_rate_30d: float
    user_chargeback_rate: float
    label: Optional[int] = None

class PredictionRequest(BaseModel):
    features: FeatureVector

class PredictionResponse(BaseModel):
    transaction_id: str
    fraud_probability: float
    decision: str
    model_version: str
    latency_ms: float
