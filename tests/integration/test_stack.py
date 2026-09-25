"""Integration tests against a running Compose stack (``docker compose up -d``).

They are skipped when the stack is not reachable, unless ``REQUIRE_STACK=1`` is set
(as in CI), in which case an unreachable stack is a failure.
"""

from __future__ import annotations

import json
import os
import socket
import time
import uuid
from typing import Any

import fakeredis
import mlflow
import numpy as np
import pytest
import redis
from confluent_kafka import Consumer, Producer
from mlflow.tracking import MlflowClient

from fraud_detection.features import FEATURE_COLUMNS, FeatureEngineer, FeatureStoreConfig
from fraud_detection.kafka_utils import consumer_config
from fraud_detection.registry import load_model, resolve_alias
from fraud_detection.schemas import DecisionEvent
from fraud_detection.simulation import DAY, SimulationConfig, TransactionSimulator

pytestmark = pytest.mark.integration

KAFKA = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5001")


def _reachable(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=2):
            return True
    except OSError:
        return False


@pytest.fixture(scope="module", autouse=True)
def stack() -> None:
    endpoints = {
        "kafka": ("localhost", 9092),
        "redis": ("localhost", 6379),
        "mlflow": ("localhost", 5001),
    }
    missing = [name for name, address in endpoints.items() if not _reachable(*address)]
    if missing:
        message = f"stack not reachable: {', '.join(missing)}"
        if os.environ.get("REQUIRE_STACK") == "1":
            pytest.fail(message)
        pytest.skip(message)


def test_real_redis_matches_the_offline_feature_store() -> None:
    """Online (real Redis) and offline (in-memory) feature computation agree."""
    prefix = f"itest-{uuid.uuid4().hex[:8]}"
    config = FeatureStoreConfig(key_prefix=prefix)
    events = list(
        TransactionSimulator(SimulationConfig(seed=5, n_users=80, n_merchants=30)).events(
            1_767_225_600.0, 1_767_225_600.0 + 4 * DAY
        )
    )
    client = redis.Redis.from_url(REDIS_URL)
    try:
        online = FeatureEngineer(client, config).compute_batch(events)
        offline = FeatureEngineer(fakeredis.FakeRedis(), config).compute_batch(events)
    finally:
        keys = list(client.scan_iter(match=f"{prefix}:*", count=1_000))
        if keys:
            client.unlink(*keys)

    online_rows = np.array(
        [[getattr(e.features, c) for c in FEATURE_COLUMNS] for e in online], float
    )
    offline_rows = np.array(
        [[getattr(e.features, c) for c in FEATURE_COLUMNS] for e in offline], float
    )
    merchant = FEATURE_COLUMNS.index("merchant_fraud_rate_30d")
    exact = [i for i in range(len(FEATURE_COLUMNS)) if i != merchant]
    np.testing.assert_array_equal(online_rows[:, exact], offline_rows[:, exact])
    # Redis HyperLogLog counts are estimates (the in-memory emulation is exact).
    np.testing.assert_allclose(online_rows[:, merchant], offline_rows[:, merchant], atol=0.02)


def _produce(topic: str, value: bytes, key: bytes = b"itest") -> None:
    producer = Producer({"bootstrap.servers": KAFKA})
    producer.produce(topic, key=key, value=value)
    assert producer.flush(10) == 0


def _produce_after_subscription(
    topic_in: str, value: bytes, topic_out: str, predicate: Any
) -> dict[str, Any]:
    """Subscribe to the output topic first, then publish, then wait for the result."""
    consumer = Consumer(
        consumer_config(KAFKA, f"itest-{uuid.uuid4().hex[:8]}", **{"auto.offset.reset": "latest"})
    )
    consumer.subscribe([topic_out])
    try:
        deadline = time.monotonic() + 30
        while not consumer.assignment() and time.monotonic() < deadline:
            consumer.poll(0.5)
        _produce(topic_in, value)
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            for msg in consumer.consume(num_messages=500, timeout=1.0):
                if msg.error() is None and msg.value():
                    payload: dict[str, Any] = json.loads(msg.value() or b"{}")
                    if predicate(payload):
                        return payload
        raise AssertionError(f"no matching message on {topic_out}")
    finally:
        consumer.close()


def test_a_transaction_flows_through_to_a_decision() -> None:
    tx_id = f"itest-{uuid.uuid4()}"
    raw = {
        "transaction_id": tx_id,
        "user_id": "u000042",
        "merchant_id": "m00007",
        "merchant_category": "electronics",
        "amount": 1_899.0,
        "country": "BR",
        "card_present": False,
        "timestamp": time.time(),
        "is_fraud": True,
        "emitted_at": time.time(),
    }

    decision = _produce_after_subscription(
        "transactions.raw",
        json.dumps(raw).encode(),
        "transactions.decisions",
        lambda payload: payload.get("transaction_id") == tx_id,
    )

    event = DecisionEvent.model_validate(decision)
    assert event.label == 1
    assert 0.0 <= event.fraud_probability <= 1.0
    assert event.features.amount == pytest.approx(1_899.0)
    assert event.features.card_present is False


def test_malformed_messages_are_dead_lettered() -> None:
    marker = f"itest-{uuid.uuid4()}"

    letter = _produce_after_subscription(
        "transactions.raw",
        json.dumps({"marker": marker, "amount": "not a number"}).encode(),
        "transactions.dlq",
        lambda payload: marker in str(payload.get("payload", "")),
    )

    assert letter["stage"] == "validate"
    assert letter["source_topic"] == "transactions.raw"


def test_champion_model_loads_from_the_registry() -> None:
    mlflow.set_tracking_uri(MLFLOW_URI)
    champion = resolve_alias(MlflowClient(), "fraud-detector", "champion")
    assert champion is not None

    model = load_model(champion.uri, champion.version)

    assert model.feature_columns == FEATURE_COLUMNS
    probabilities = model.scorer.predict_proba(np.zeros((3, len(FEATURE_COLUMNS))))
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
