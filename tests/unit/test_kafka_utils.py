from __future__ import annotations

import json
import logging
from typing import Any

import pytest
from confluent_kafka import KafkaError, KafkaException

from fraud_detection.kafka_utils import (
    DeliveryError,
    DeliveryTracker,
    GracefulShutdown,
    build_dead_letter,
    check_message_error,
    consumer_config,
    ensure_topics,
    producer_config,
    retry_call,
    store_offsets,
)


def test_store_offsets_commits_next_offset_per_partition(kafka: Any) -> None:
    consumer = kafka.Consumer([])
    messages = [
        kafka.Message(b"a", partition=0, offset=5),
        kafka.Message(b"b", partition=0, offset=7),
        kafka.Message(b"c", partition=1, offset=3),
        kafka.Message(None, partition=1, offset=9, error=kafka.Error(1)),
    ]

    store_offsets(consumer, messages)

    stored = {(tp.topic, tp.partition): tp.offset for tp in consumer.stored}
    assert stored == {("transactions.raw", 0): 8, ("transactions.raw", 1): 4}


def test_store_offsets_ignores_empty_batches(kafka: Any) -> None:
    consumer = kafka.Consumer([])
    store_offsets(consumer, [])
    assert consumer.stored == []


class TestCheckMessageError:
    def test_data_message(self, kafka: Any) -> None:
        assert check_message_error(kafka.Message(b"x")) is True

    def test_non_fatal_errors_are_skipped(
        self, kafka: Any, caplog: pytest.LogCaptureFixture
    ) -> None:
        error = kafka.Error(KafkaError.UNKNOWN_TOPIC_OR_PART, name="UNKNOWN_TOPIC_OR_PART")
        with caplog.at_level(logging.WARNING):
            assert check_message_error(kafka.Message(None, error=error)) is False
        assert "consumer_error" in caplog.text

    def test_partition_eof_is_silent(self, kafka: Any, caplog: pytest.LogCaptureFixture) -> None:
        eof = kafka.Message(None, error=kafka.Error(KafkaError._PARTITION_EOF))
        with caplog.at_level(logging.WARNING):
            assert check_message_error(eof) is False
        assert caplog.text == ""

    def test_fatal_errors_raise(self, kafka: Any) -> None:
        with pytest.raises(KafkaException):
            check_message_error(kafka.Message(None, error=kafka.Error(1, fatal=True)))


def test_dead_letter_keeps_origin_and_payload(kafka: Any) -> None:
    message = kafka.Message(b'{"bad": 1}', key=b"u1", partition=2, offset=42)

    letter = json.loads(build_dead_letter(message, ValueError("boom"), "validate"))

    assert letter["stage"] == "validate"
    assert letter["error"] == "ValueError: boom"
    assert (letter["source_topic"], letter["partition"], letter["offset"]) == (
        "transactions.raw",
        2,
        42,
    )
    assert letter["key"] == "u1"
    assert letter["payload"] == '{"bad": 1}'


def test_dead_letter_tolerates_binary_garbage(kafka: Any) -> None:
    letter = json.loads(build_dead_letter(kafka.Message(b"\xff\xfe"), ValueError("x"), "validate"))
    assert letter["payload"] == "��"


def test_delivery_tracker_raises_once(kafka: Any) -> None:
    tracker = DeliveryTracker()
    tracker(None, kafka.Message(b"ok"))
    tracker(kafka.Error(1), kafka.Message(b"bad", topic="out"))

    with pytest.raises(DeliveryError, match="1 deliveries failed"):
        tracker.raise_for_failures()
    tracker.raise_for_failures()  # reset after raising


class TestRetryCall:
    def test_retries_until_success(self, shutdown: GracefulShutdown) -> None:
        calls = {"n": 0}

        def flaky() -> str:
            calls["n"] += 1
            if calls["n"] < 3:
                raise ConnectionError("down")
            return "ok"

        result = retry_call(
            flaky,
            retry_on=(ConnectionError,),
            shutdown=shutdown,
            description="test",
            max_backoff_s=0.01,
        )
        assert result == "ok"

    def test_gives_up_after_max_elapsed(self, shutdown: GracefulShutdown) -> None:
        def always_down() -> None:
            raise ConnectionError("down")

        with pytest.raises(ConnectionError):
            retry_call(
                always_down,
                retry_on=(ConnectionError,),
                shutdown=shutdown,
                description="test",
                max_backoff_s=0.01,
                max_elapsed_s=0.05,
            )

    def test_stops_on_shutdown(self, shutdown: GracefulShutdown) -> None:
        shutdown.request()

        def always_down() -> None:
            raise ConnectionError("down")

        with pytest.raises(ConnectionError):
            retry_call(always_down, retry_on=(ConnectionError,), shutdown=shutdown, description="t")

    def test_does_not_retry_other_errors(self, shutdown: GracefulShutdown) -> None:
        def bug() -> None:
            raise KeyError("bug")

        with pytest.raises(KeyError):
            retry_call(bug, retry_on=(ConnectionError,), shutdown=shutdown, description="t")


class FakeFuture:
    def __init__(self, error: Exception | None = None) -> None:
        self._error = error

    def result(self) -> None:
        if self._error:
            raise self._error


class FakeAdmin:
    def __init__(self, existing: set[str], failures_before_ready: int = 0) -> None:
        self.existing = existing
        self.failures_before_ready = failures_before_ready
        self.created: list[str] = []

    def list_topics(self, timeout: float) -> Any:
        if self.failures_before_ready:
            self.failures_before_ready -= 1
            raise KafkaException(KafkaError(KafkaError._TRANSPORT))
        return type("Metadata", (), {"topics": dict.fromkeys(self.existing)})()

    def create_topics(self, topics: list[Any]) -> dict[str, FakeFuture]:
        futures = {}
        for topic in topics:
            self.created.append(topic.topic)
            if topic.topic == "raced":
                futures[topic.topic] = FakeFuture(
                    KafkaException(KafkaError(KafkaError.TOPIC_ALREADY_EXISTS))
                )
            else:
                futures[topic.topic] = FakeFuture()
        return futures


def test_ensure_topics_creates_only_missing_topics() -> None:
    admin = FakeAdmin(existing={"a"})

    ensure_topics(admin, {"a": 3, "b": 3, "raced": 1}, replication_factor=1)  # type: ignore[arg-type]

    assert admin.created == ["b", "raced"]


def test_ensure_topics_waits_for_the_broker(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("fraud_detection.kafka_utils.time.sleep", lambda _: None)
    admin = FakeAdmin(existing=set(), failures_before_ready=2)

    ensure_topics(admin, {"a": 1}, replication_factor=1)  # type: ignore[arg-type]

    assert admin.created == ["a"]


def test_client_configs_enforce_at_least_once_defaults() -> None:
    consumer = consumer_config("broker:9092", "group", **{"auto.offset.reset": "latest"})
    producer = producer_config("broker:9092")

    assert consumer["enable.auto.offset.store"] is False
    assert consumer["auto.offset.reset"] == "latest"
    assert producer["enable.idempotence"] is True
    assert producer["acks"] == "all"


def test_graceful_shutdown_handles_signals() -> None:
    import signal

    previous = signal.getsignal(signal.SIGUSR1)
    try:
        shutdown = GracefulShutdown(signals=(signal.SIGUSR1,))
        before = shutdown.requested
        signal.raise_signal(signal.SIGUSR1)
        after = shutdown.requested
        assert (before, after) == (False, True)
        assert shutdown.wait(0) is True
    finally:
        signal.signal(signal.SIGUSR1, previous)
