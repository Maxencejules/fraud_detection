"""Event and API schemas shared by every service.

``FeatureVector`` is the model input contract: its field order *is* the feature order
used for training and inference (see ``fraud_detection.features.FEATURE_COLUMNS``).
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

Identifier = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_.:-]{1,64}$")]
CountryCode = Annotated[str, StringConstraints(pattern=r"^[A-Z]{2}$")]
CurrencyCode = Annotated[str, StringConstraints(pattern=r"^[A-Z]{3}$")]
# The upper bound is the end of year 9999, where datetime's range ends. Larger values are
# almost always milliseconds or microseconds; rejecting them at validation keeps a unit
# mistake from reaching the feature store, where it would trim the user's entire history.
MAX_UNIX_SECONDS = 253_402_300_800.0
UnixSeconds = Annotated[
    float,
    Field(gt=0.0, lt=MAX_UNIX_SECONDS, description="Seconds since the Unix epoch (UTC)."),
]


class Decision(StrEnum):
    APPROVE = "APPROVE"
    REVIEW = "REVIEW"
    BLOCK = "BLOCK"


class RawTransaction(BaseModel):
    """A card authorisation request as published on the raw transactions topic."""

    model_config = ConfigDict(frozen=True, extra="ignore", allow_inf_nan=False)

    transaction_id: Identifier
    user_id: Identifier
    merchant_id: Identifier
    merchant_category: Annotated[str, StringConstraints(min_length=1, max_length=32)]
    amount: float = Field(gt=0.0, le=1_000_000.0)
    currency: CurrencyCode = "USD"
    country: CountryCode
    card_present: bool
    timestamp: UnixSeconds = Field(description="Event time of the authorisation.")
    device_id: Annotated[str, StringConstraints(max_length=64)] | None = None
    is_fraud: bool | None = Field(
        default=None,
        description=(
            "Ground truth attached by the simulator. A production system would receive "
            "labels later through a separate chargeback feed."
        ),
    )
    emitted_at: UnixSeconds | None = Field(
        default=None, description="Wall-clock time the producer published the event."
    )


class FeatureVector(BaseModel):
    """Model input features. Field order defines the model's column order."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    amount: float = Field(ge=0.0, le=1_000_000.0)
    amount_log: float = Field(ge=0.0, le=20.0)
    amount_zscore: float = Field(ge=-1_000.0, le=1_000.0)
    tx_count_1h: int = Field(ge=0)
    tx_count_24h: int = Field(ge=0)
    tx_sum_1h: float = Field(ge=0.0)
    tx_sum_24h: float = Field(ge=0.0)
    unique_merchants_24h: int = Field(ge=0)
    unique_countries_7d: int = Field(ge=0)
    seconds_since_last_tx: float = Field(ge=0.0)
    is_new_country: bool
    hour_of_day: int = Field(ge=0, le=23)
    day_of_week: int = Field(ge=0, le=6)
    is_weekend: bool
    card_present: bool
    merchant_fraud_rate_30d: float = Field(ge=0.0, le=1.0)
    user_chargeback_rate: float = Field(ge=0.0, le=1.0)


class FeatureEvent(BaseModel):
    """Feature vector for one transaction, published on the features topic."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    transaction_id: Identifier
    user_id: Identifier
    merchant_id: Identifier
    event_time: UnixSeconds
    emitted_at: UnixSeconds | None = None
    processed_at: UnixSeconds
    label: int | None = Field(default=None, ge=0, le=1)
    features: FeatureVector


class PredictionRequest(FeatureVector):
    transaction_id: Identifier


class BatchPredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[PredictionRequest] = Field(min_length=1)


class ScoredTransaction(BaseModel):
    transaction_id: str
    fraud_probability: float = Field(ge=0.0, le=1.0)
    decision: Decision


class PredictionResponse(ScoredTransaction):
    model_config = ConfigDict(protected_namespaces=())

    model_version: str
    latency_ms: float


class BatchPredictionResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    items: list[ScoredTransaction]
    model_version: str
    latency_ms: float


class DecisionEvent(BaseModel):
    """A scored transaction, published on the decisions topic."""

    model_config = ConfigDict(frozen=True, extra="ignore", protected_namespaces=())

    transaction_id: Identifier
    user_id: Identifier
    merchant_id: Identifier
    event_time: UnixSeconds
    emitted_at: UnixSeconds | None = None
    scored_at: UnixSeconds
    fraud_probability: float = Field(ge=0.0, le=1.0)
    decision: Decision
    model_version: str
    label: int | None = Field(default=None, ge=0, le=1)
    features: FeatureVector
