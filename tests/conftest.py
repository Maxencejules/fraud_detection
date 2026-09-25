from __future__ import annotations

import itertools
from collections.abc import Callable, Iterator
from types import SimpleNamespace
from typing import Any

import fakeredis
import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from fraud_detection.features import FEATURE_COLUMNS, FeatureEngineer, FeatureStoreConfig
from fraud_detection.kafka_utils import GracefulShutdown
from fraud_detection.modeling import EnsembleScorer, LogitCalibrator
from fraud_detection.schemas import RawTransaction

T0 = 1_767_225_600.0  # 2026-01-01T00:00:00Z, a Thursday

TxFactory = Callable[..., RawTransaction]


@pytest.fixture
def redis_client() -> Iterator[fakeredis.FakeRedis]:
    client = fakeredis.FakeRedis()
    yield client
    client.flushall()


@pytest.fixture
def engineer(redis_client: fakeredis.FakeRedis) -> FeatureEngineer:
    return FeatureEngineer(redis_client, FeatureStoreConfig())


@pytest.fixture
def make_tx() -> TxFactory:
    counter = itertools.count()

    def factory(**overrides: Any) -> RawTransaction:
        fields: dict[str, Any] = {
            "transaction_id": f"tx-{next(counter):06d}",
            "user_id": "u000001",
            "merchant_id": "m00001",
            "merchant_category": "retail",
            "amount": 50.0,
            "country": "US",
            "card_present": True,
            "timestamp": T0,
            "is_fraud": False,
        }
        fields.update(overrides)
        return RawTransaction(**fields)

    return factory


@pytest.fixture(scope="session")
def tiny_scorer() -> EnsembleScorer:
    """A real (tiny) ensemble trained on random data, for fast inference tests."""
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(rng.random((400, len(FEATURE_COLUMNS))), columns=list(FEATURE_COLUMNS))
    labels = (frame["amount_zscore"] + 0.3 * rng.random(400) > 0.9).astype(int)
    xgb_model = xgb.XGBClassifier(n_estimators=10, max_depth=2).fit(frame, labels)
    lgb_model = lgb.LGBMClassifier(
        n_estimators=10, num_leaves=4, min_child_samples=5, verbose=-1
    ).fit(frame, labels)
    return EnsembleScorer(
        xgb_model.get_booster(), lgb_model.booster_, LogitCalibrator.identity(), FEATURE_COLUMNS
    )


PayloadFactory = Callable[..., dict[str, Any]]


@pytest.fixture
def feature_payload() -> PayloadFactory:
    """Builds a valid feature payload; keyword arguments override fields."""
    return _feature_payload


def _feature_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "amount": 42.0,
        "amount_log": 3.76,
        "amount_zscore": 0.2,
        "tx_count_1h": 1,
        "tx_count_24h": 3,
        "tx_sum_1h": 42.0,
        "tx_sum_24h": 120.0,
        "unique_merchants_24h": 2,
        "unique_countries_7d": 1,
        "seconds_since_last_tx": 3_600.0,
        "is_new_country": False,
        "hour_of_day": 14,
        "day_of_week": 2,
        "is_weekend": False,
        "card_present": True,
        "merchant_fraud_rate_30d": 0.01,
        "user_chargeback_rate": 0.0,
    }
    payload.update(overrides)
    return payload


# -- Kafka test doubles --------------------------------------------------------------


class FakeKafkaError:
    def __init__(self, code: int, fatal: bool = False, name: str = "ERR") -> None:
        self._code, self._fatal, self._name = code, fatal, name

    def code(self) -> int:
        return self._code

    def fatal(self) -> bool:
        return self._fatal

    def name(self) -> str:
        return self._name

    def str(self) -> str:
        return f"{self._name} ({self._code})"


class FakeMessage:
    def __init__(
        self,
        value: bytes | None,
        *,
        key: bytes | None = b"k",
        topic: str = "transactions.raw",
        partition: int = 0,
        offset: int = 0,
        error: FakeKafkaError | None = None,
    ) -> None:
        self._value, self._key, self._topic = value, key, topic
        self._partition, self._offset, self._error = partition, offset, error

    def value(self) -> bytes | None:
        return self._value

    def key(self) -> bytes | None:
        return self._key

    def topic(self) -> str:
        return self._topic

    def partition(self) -> int:
        return self._partition

    def offset(self) -> int:
        return self._offset

    def error(self) -> FakeKafkaError | None:
        return self._error


class FakeProducer:
    """Records produced messages; delivery reports fire on ``poll``/``flush``."""

    def __init__(self) -> None:
        self.messages: list[tuple[str, Any, Any]] = []
        self.failing_topics: set[str] = set()
        self.buffer_errors = 0
        self._pending: list[tuple[Any, str]] = []

    def produce(
        self, topic: str, key: Any = None, value: Any = None, on_delivery: Any = None
    ) -> None:
        if self.buffer_errors:
            self.buffer_errors -= 1
            raise BufferError("queue full")
        self.messages.append((topic, key, value))
        if on_delivery is not None:
            self._pending.append((on_delivery, topic))

    def _deliver(self) -> int:
        pending, self._pending = self._pending, []
        for callback, topic in pending:
            error = FakeKafkaError(1, name="DELIVERY") if topic in self.failing_topics else None
            callback(error, FakeMessage(None, topic=topic))
        return len(pending)

    def poll(self, timeout: float = 0) -> int:
        return self._deliver()

    def flush(self, timeout: float | None = None) -> int:
        self._deliver()
        return 0

    def values(self, topic: str) -> list[Any]:
        return [value for t, _, value in self.messages if t == topic]


class FakeConsumer:
    """Serves pre-baked batches, then asks the service to shut down."""

    def __init__(self, batches: list[list[FakeMessage]], shutdown: Any = None) -> None:
        self.batches = list(batches)
        self.shutdown = shutdown
        self.stored: list[Any] = []
        self.subscribed: list[str] = []
        self.closed = False

    def subscribe(self, topics: list[str]) -> None:
        self.subscribed = topics

    def consume(self, num_messages: int = 1, timeout: float = -1) -> list[FakeMessage]:
        if self.batches:
            return self.batches.pop(0)
        if self.shutdown is not None:
            self.shutdown.request()
        return []

    def store_offsets(self, offsets: list[Any]) -> None:
        self.stored.extend(offsets)

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def shutdown() -> GracefulShutdown:
    return GracefulShutdown(signals=())


@pytest.fixture
def kafka() -> SimpleNamespace:
    """Access to the Kafka doubles without importing from conftest."""
    return SimpleNamespace(
        Message=FakeMessage, Error=FakeKafkaError, Producer=FakeProducer, Consumer=FakeConsumer
    )
