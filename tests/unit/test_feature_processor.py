from __future__ import annotations

import json
from typing import Any

import fakeredis
import pytest
import redis
from confluent_kafka import KafkaException

from fraud_detection.config import FeatureProcessorSettings
from fraud_detection.feature_processor import FeatureProcessor
from fraud_detection.features import FeatureEngineer
from fraud_detection.kafka_utils import DeliveryError, GracefulShutdown
from fraud_detection.schemas import FeatureEvent

SETTINGS = FeatureProcessorSettings()


def _processor(
    kafka: Any, shutdown: GracefulShutdown, batches: list[Any], engineer: Any = None
) -> tuple[FeatureProcessor, Any, Any]:
    """Returns the processor plus its fake consumer and producer for assertions."""
    consumer, producer = kafka.Consumer(batches, shutdown), kafka.Producer()
    processor = FeatureProcessor(
        consumer=consumer,
        producer=producer,
        engineer=engineer or FeatureEngineer(fakeredis.FakeRedis()),
        settings=SETTINGS,
        shutdown=shutdown,
    )
    return processor, consumer, producer


def _raw(make_tx: Any, **overrides: Any) -> bytes:
    payload: str = make_tx(**overrides).model_dump_json()
    return payload.encode()


def test_valid_batch_is_published_and_committed(
    kafka: Any, shutdown: GracefulShutdown, make_tx: Any
) -> None:
    batch = [
        kafka.Message(_raw(make_tx, user_id="u1"), offset=0),
        kafka.Message(_raw(make_tx, user_id="u2"), offset=1),
    ]
    processor, consumer, producer = _processor(kafka, shutdown, [batch])

    processor.process_batch(batch)

    published = [
        FeatureEvent.model_validate_json(v) for v in producer.values(SETTINGS.topic_features)
    ]
    assert [e.user_id for e in published] == ["u1", "u2"]
    keys = [k for t, k, _ in producer.messages if t == SETTINGS.topic_features]
    assert keys == ["u1", "u2"]  # keyed by user so a user's events stay ordered
    assert [tp.offset for tp in consumer.stored] == [2]


def test_invalid_messages_go_to_the_dead_letter_topic(
    kafka: Any, shutdown: GracefulShutdown, make_tx: Any
) -> None:
    batch = [
        kafka.Message(b"not json", offset=0),
        kafka.Message(json.dumps({"transaction_id": "t"}).encode(), offset=1),
        kafka.Message(None, offset=2),  # tombstone
        kafka.Message(_raw(make_tx), offset=3),
    ]
    processor, consumer, producer = _processor(kafka, shutdown, [batch])

    processor.process_batch(batch)

    letters = [json.loads(v) for v in producer.values(SETTINGS.topic_dlq)]
    assert [(letter["offset"], letter["stage"]) for letter in letters] == [
        (0, "validate"),
        (1, "validate"),
        (2, "validate"),
    ]
    assert len(producer.values(SETTINGS.topic_features)) == 1
    assert [tp.offset for tp in consumer.stored] == [4]


def test_timestamps_in_the_wrong_unit_never_reach_the_feature_store(
    kafka: Any, shutdown: GracefulShutdown, make_tx: Any
) -> None:
    client = fakeredis.FakeRedis()
    engineer = FeatureEngineer(client)
    history = make_tx(user_id="u1")
    engineer.compute_batch([history])
    in_milliseconds = make_tx(user_id="u1").model_dump(mode="json")
    in_milliseconds["timestamp"] = (history.timestamp + 60) * 1000
    batch = [kafka.Message(json.dumps(in_milliseconds).encode(), offset=0)]
    processor, consumer, producer = _processor(kafka, shutdown, [batch], engineer=engineer)

    processor.process_batch(batch)

    letters = [json.loads(v) for v in producer.values(SETTINGS.topic_dlq)]
    assert [letter["stage"] for letter in letters] == ["validate"]
    assert client.zcard(engineer.config.user_key("u1", "tx")) == 1  # history untouched
    assert [tp.offset for tp in consumer.stored] == [1]


def test_offsets_are_not_committed_when_delivery_fails(
    kafka: Any, shutdown: GracefulShutdown, make_tx: Any
) -> None:
    batch = [kafka.Message(_raw(make_tx))]
    processor, consumer, producer = _processor(kafka, shutdown, [batch])
    producer.failing_topics.add(SETTINGS.topic_features)

    with pytest.raises(DeliveryError):
        processor.process_batch(batch)
    assert consumer.stored == []


def test_redis_outages_are_retried_not_dead_lettered(
    kafka: Any, shutdown: GracefulShutdown, make_tx: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("fraud_detection.kafka_utils.GracefulShutdown.wait", lambda *_: False)
    engineer = FeatureEngineer(fakeredis.FakeRedis())
    real_compute = engineer.compute_batch
    calls = {"n": 0}

    def flaky(txs: Any) -> Any:
        calls["n"] += 1
        if calls["n"] == 1:
            raise redis.ConnectionError("redis down")
        return real_compute(txs)

    monkeypatch.setattr(engineer, "compute_batch", flaky)
    batch = [kafka.Message(_raw(make_tx))]
    processor, _consumer, producer = _processor(kafka, shutdown, [batch], engineer)

    processor.process_batch(batch)

    assert calls["n"] == 2
    assert producer.values(SETTINGS.topic_dlq) == []
    assert len(producer.values(SETTINGS.topic_features)) == 1


def test_transient_consumer_errors_are_skipped(
    kafka: Any, shutdown: GracefulShutdown, make_tx: Any
) -> None:
    batch = [
        kafka.Message(None, offset=0, error=kafka.Error(3, name="UNKNOWN_TOPIC_OR_PART")),
        kafka.Message(_raw(make_tx), offset=1),
    ]
    processor, consumer, producer = _processor(kafka, shutdown, [batch])

    processor.process_batch(batch)

    assert len(producer.values(SETTINGS.topic_features)) == 1
    assert [tp.offset for tp in consumer.stored] == [2]


def test_fatal_consumer_errors_stop_processing(kafka: Any, shutdown: GracefulShutdown) -> None:
    batch = [kafka.Message(None, error=kafka.Error(1, fatal=True))]
    processor, _consumer, _producer = _processor(kafka, shutdown, [batch])

    with pytest.raises(KafkaException):
        processor.process_batch(batch)


def test_run_drains_and_closes_on_shutdown(
    kafka: Any, shutdown: GracefulShutdown, make_tx: Any
) -> None:
    batches = [[kafka.Message(_raw(make_tx), offset=i)] for i in range(3)]
    processor, consumer, producer = _processor(kafka, shutdown, batches)

    processor.run()

    assert consumer.subscribed == [SETTINGS.topic_raw]
    assert len(producer.values(SETTINGS.topic_features)) == 3
    assert consumer.closed is True
