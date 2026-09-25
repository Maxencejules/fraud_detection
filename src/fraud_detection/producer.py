"""Publishes simulated card transactions to Kafka at a steady rate.

Events carry simulated *event time* (``timestamp``) plus the wall-clock time they were
published (``emitted_at``), which downstream services use to measure end-to-end latency.
The simulation clock is checkpointed in Redis so a restart continues the timeline instead
of replaying it. A checkpoint is written only after Kafka has acknowledged every event up
to it, so a crash or a failed delivery never leaves a gap: the restart resumes from the
last acknowledged event time.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterable

import redis
from confluent_kafka import KafkaError, Message, Producer
from confluent_kafka.admin import AdminClient
from prometheus_client import Counter, Gauge

from fraud_detection.config import ProducerSettings
from fraud_detection.kafka_utils import (
    DeliveryError,
    DeliveryTracker,
    GracefulShutdown,
    ensure_topics,
    producer_config,
)
from fraud_detection.observability import configure_logging, start_metrics_server
from fraud_detection.schemas import RawTransaction
from fraud_detection.simulation import TransactionSimulator

logger = logging.getLogger("fraud_detection.producer")

EVENTS = Counter(
    "fraud_producer_events_total", "Transactions published, by ground truth.", ["kind"]
)
DELIVERY_FAILURES = Counter("fraud_producer_delivery_failures_total", "Unacknowledged messages.")
SIMULATED_TIME = Gauge("fraud_producer_simulated_time_seconds", "Event time of the last message.")

CLOCK_SAVE_INTERVAL_S = 1.0
CHECKPOINT_FLUSH_TIMEOUT_S = 10.0
FINAL_FLUSH_TIMEOUT_S = 30.0


class Pacer:
    """Spaces calls ``1 / rate`` seconds apart on an absolute schedule (no drift)."""

    def __init__(self, rate_per_s: float, clock: Callable[[], float] = time.monotonic) -> None:
        self._interval = 1.0 / rate_per_s
        self._clock = clock
        self._next = clock()

    def delay(self) -> float:
        """Seconds to wait before the next event; skips ahead rather than bursting when late."""
        now = self._clock()
        if now - self._next > 1.0:
            self._next = now
        wait = max(0.0, self._next - now)
        self._next += self._interval
        return wait


def publish(
    producer: Producer,
    topic: str,
    transactions: Iterable[RawTransaction],
    *,
    rate_per_s: float,
    shutdown: GracefulShutdown,
    save_clock: Callable[[float], None],
    checkpoint_interval_s: float = CLOCK_SAVE_INTERVAL_S,
) -> int:
    """Publish ``transactions`` until the iterator ends or shutdown is requested.

    Every ``checkpoint_interval_s`` the producer flushes and saves the event time of the
    newest event, but only when every event so far has been acknowledged. Raises
    :class:`DeliveryError`, without advancing the clock, when a delivery fails or events
    are still unacknowledged at the end.
    """
    pacer = Pacer(rate_per_s)
    tracker = DeliveryTracker()

    def on_delivery(err: KafkaError | None, msg: Message) -> None:
        if err is not None:
            DELIVERY_FAILURES.inc()
            logger.error("delivery_failed", extra={"error": str(err)})
        tracker(err, msg)

    def checkpoint(event_time: float, timeout_s: float) -> int:
        """Save ``event_time`` if everything is acknowledged; return the count still queued."""
        remaining = producer.flush(timeout_s)
        tracker.raise_for_failures()
        if remaining == 0:
            save_clock(event_time)
        return remaining

    published = 0
    last_event_time = 0.0
    unsaved = False  # events produced since the last saved checkpoint
    last_checkpoint = time.monotonic()
    for tx in transactions:
        if shutdown.wait(pacer.delay()):
            break
        event = tx.model_copy(update={"emitted_at": time.time()})
        while True:
            try:
                producer.produce(
                    topic, key=tx.user_id, value=event.model_dump_json(), on_delivery=on_delivery
                )
                break
            except BufferError:  # local queue full: let the client drain it
                producer.poll(0.5)
        producer.poll(0)
        published += 1
        last_event_time = tx.timestamp
        unsaved = True
        EVENTS.labels(kind="fraud" if tx.is_fraud else "legit").inc()
        SIMULATED_TIME.set(tx.timestamp)
        if time.monotonic() - last_checkpoint >= checkpoint_interval_s:
            queued = checkpoint(tx.timestamp, CHECKPOINT_FLUSH_TIMEOUT_S)
            if queued:  # broker slow or unreachable: keep the previous checkpoint
                logger.warning("checkpoint_deferred", extra={"queued": queued})
            else:
                unsaved = False
            last_checkpoint = time.monotonic()
    if unsaved:
        queued = checkpoint(last_event_time, FINAL_FLUSH_TIMEOUT_S)
        if queued:
            raise DeliveryError(f"{queued} messages still queued after flush; clock not saved")
    return published


def main() -> None:
    settings = ProducerSettings()
    configure_logging("producer", settings.log_level)
    start_metrics_server(settings.metrics_port, logger)
    shutdown = GracefulShutdown()

    clock_key = settings.feature_store_config().clock_key()
    store = redis.Redis.from_url(settings.redis_url)

    def save_clock(event_time: float) -> None:
        try:
            store.set(clock_key, repr(event_time))
        except redis.RedisError as exc:  # best effort: only affects restart continuity
            logger.warning("clock_save_failed", extra={"error": str(exc)})

    saved = store.get(clock_key)
    start = float(saved) if saved else time.time()
    logger.info(
        "producer_starting",
        extra={
            "simulated_start": start,
            "resumed": saved is not None,
            "rate_tps": settings.emit_rate_tps,
        },
    )

    admin = AdminClient({"bootstrap.servers": settings.kafka_bootstrap_servers})
    ensure_topics(
        admin, {settings.topic_raw: settings.topic_partitions}, settings.topic_replication_factor
    )
    producer = Producer(producer_config(settings.kafka_bootstrap_servers))
    simulator = TransactionSimulator(settings.simulation_config(), stream=f"live-{start!r}")
    published = publish(
        producer,
        settings.topic_raw,
        simulator.events(start),
        rate_per_s=settings.emit_rate_tps,
        shutdown=shutdown,
        save_clock=save_clock,
    )
    logger.info("producer_stopped", extra={"published": published})


if __name__ == "__main__":
    main()
