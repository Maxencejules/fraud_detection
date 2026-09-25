"""Kafka plumbing shared by the streaming services.

Delivery semantics used throughout the pipeline (at-least-once):

1. consume a batch with ``enable.auto.offset.store=false``;
2. produce every output (including dead letters) for the batch;
3. ``flush`` the producer and fail if any delivery failed;
4. only then store the batch's offsets, which the consumer auto-commits.

A crash anywhere before step 4 replays the batch. Downstream processing is idempotent
(feature state) or keyed by ``transaction_id`` (decisions), so replays are harmless.
"""

from __future__ import annotations

import json
import logging
import signal
import threading
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import UTC, datetime
from types import FrameType
from typing import Any

from confluent_kafka import KafkaError, KafkaException, Message, TopicPartition
from confluent_kafka.admin import (  # type: ignore[attr-defined]  # documented path, no re-export
    AdminClient,
    NewTopic,
)

logger = logging.getLogger("fraud_detection.kafka")


class GracefulShutdown:
    """Turns SIGTERM/SIGINT into a flag so loops can drain and commit before exiting.

    Docker stops containers with SIGTERM; without a handler Python dies immediately and
    in-flight batches are neither flushed nor committed.
    """

    def __init__(self, signals: Sequence[signal.Signals] = (signal.SIGTERM, signal.SIGINT)):
        self._event = threading.Event()
        for sig in signals:
            signal.signal(sig, self._handle)

    def _handle(self, signum: int, _frame: FrameType | None) -> None:
        logger.info("shutdown_requested", extra={"signal": signal.Signals(signum).name})
        self._event.set()

    @property
    def requested(self) -> bool:
        return self._event.is_set()

    def request(self) -> None:
        self._event.set()

    def wait(self, timeout: float) -> bool:
        """Sleep up to ``timeout`` seconds; returns early (True) when shutdown is requested."""
        return self._event.wait(timeout)


class DeliveryError(RuntimeError):
    """At least one produced message was not acknowledged by the broker."""


class DeliveryTracker:
    """Collects asynchronous delivery reports for one batch."""

    def __init__(self) -> None:
        self.failures: list[str] = []

    def __call__(self, err: KafkaError | None, msg: Message) -> None:
        if err is not None:
            self.failures.append(f"{msg.topic()}: {err}")

    def raise_for_failures(self) -> None:
        if self.failures:
            failures, self.failures = self.failures, []
            raise DeliveryError(f"{len(failures)} deliveries failed, first: {failures[0]}")


def ensure_topics(
    admin: AdminClient,
    topics: Mapping[str, int],
    replication_factor: int,
    timeout_s: float = 60.0,
) -> None:
    """Create missing topics (name -> partitions). Retries while the broker starts up."""
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            existing = set(admin.list_topics(timeout=10).topics)
            missing = [
                NewTopic(name, num_partitions=partitions, replication_factor=replication_factor)
                for name, partitions in topics.items()
                if name not in existing
            ]
            if missing:
                for name, future in admin.create_topics(missing).items():
                    try:
                        future.result()
                        logger.info("topic_created", extra={"topic": name})
                    except KafkaException as exc:
                        if exc.args[0].code() != KafkaError.TOPIC_ALREADY_EXISTS:
                            raise
            return
        except KafkaException:
            if time.monotonic() > deadline:
                raise
            logger.warning("kafka_not_ready", extra={"retry_in_s": 2})
            time.sleep(2)


def check_message_error(msg: Message) -> bool:
    """Return True if ``msg`` carries data. Raises on fatal errors, logs the others.

    librdkafka reports transient conditions (broker restarts, a subscribed topic that
    does not exist *yet*) as consumer errors; they must not stop the service.
    """
    err = msg.error()
    if err is None:
        return True
    if err.fatal():
        raise KafkaException(err)
    if err.code() != KafkaError._PARTITION_EOF:
        logger.warning("consumer_error", extra={"error": err.str(), "code": err.name()})
    return False


def store_offsets(consumer: Any, messages: Iterable[Message]) -> None:
    """Mark ``messages`` as processed (highest offset per partition)."""
    highest: dict[tuple[str, int], int] = {}
    for msg in messages:
        topic, partition, offset = msg.topic(), msg.partition(), msg.offset()
        if (
            msg.error() is None
            and topic is not None
            and partition is not None
            and offset is not None
        ):
            highest[(topic, partition)] = max(highest.get((topic, partition), -1), offset)
    if highest:
        consumer.store_offsets(
            offsets=[TopicPartition(t, p, offset + 1) for (t, p), offset in highest.items()]
        )


def build_dead_letter(msg: Message, error: Exception, stage: str) -> bytes:
    """Wrap a poison message with enough context to replay or debug it."""
    value = msg.value()
    key = msg.key()
    payload = {
        "failed_at": datetime.now(tz=UTC).isoformat(),
        "stage": stage,
        "error": f"{type(error).__name__}: {error}",
        "source_topic": msg.topic(),
        "partition": msg.partition(),
        "offset": msg.offset(),
        "key": key.decode("utf-8", "replace") if isinstance(key, bytes) else key,
        "payload": value.decode("utf-8", "replace") if isinstance(value, bytes) else value,
    }
    return json.dumps(payload).encode("utf-8")


def consumer_config(bootstrap_servers: str, group_id: str, **overrides: Any) -> dict[str, Any]:
    config: dict[str, Any] = {
        "bootstrap.servers": bootstrap_servers,
        "group.id": group_id,
        "auto.offset.reset": "earliest",
        "enable.auto.commit": True,
        "enable.auto.offset.store": False,
        "partition.assignment.strategy": "cooperative-sticky",
    }
    config.update(overrides)
    return config


def producer_config(bootstrap_servers: str, **overrides: Any) -> dict[str, Any]:
    config: dict[str, Any] = {
        "bootstrap.servers": bootstrap_servers,
        "enable.idempotence": True,
        "acks": "all",
        "linger.ms": 5,
        "compression.type": "lz4",
    }
    config.update(overrides)
    return config


def retry_call[T](
    operation: Callable[[], T],
    *,
    retry_on: tuple[type[BaseException], ...],
    shutdown: GracefulShutdown,
    description: str,
    max_backoff_s: float = 30.0,
    max_elapsed_s: float | None = None,
) -> T:
    """Call ``operation`` with capped exponential backoff on ``retry_on`` errors.

    Gives up (re-raises) when shutdown is requested or ``max_elapsed_s`` is exceeded, so
    an unprocessed batch is never committed.
    """
    delay = 0.5
    started = time.monotonic()
    while True:
        try:
            return operation()
        except retry_on as exc:
            elapsed = time.monotonic() - started
            if shutdown.requested or (max_elapsed_s is not None and elapsed > max_elapsed_s):
                raise
            logger.warning(
                "operation_retry",
                extra={"operation": description, "error": str(exc), "retry_in_s": delay},
            )
            shutdown.wait(delay)
            delay = min(delay * 2, max_backoff_s)
