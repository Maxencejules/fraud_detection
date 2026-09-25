"""Streams raw transactions into feature events.

For each consumed batch: validate, compute features in one Redis round trip, publish
feature events, flush, and only then store offsets (at-least-once; see ``kafka_utils``).

Error handling separates *data* errors from *infrastructure* errors:

* a malformed or invalid message is published to the dead-letter topic with its origin
  and the reason, then skipped;
* Redis or Kafka outages are retried with backoff and never dead-lettered, so valid
  transactions are not lost to the DLQ during an incident. If retries are exhausted
  the process exits without committing and the orchestrator restarts it.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence

import redis
from confluent_kafka import Consumer, Message, Producer
from confluent_kafka.admin import AdminClient
from prometheus_client import Counter, Histogram
from pydantic import ValidationError

from fraud_detection.config import FeatureProcessorSettings
from fraud_detection.features import FeatureEngineer
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
from fraud_detection.observability import configure_logging, start_metrics_server
from fraud_detection.schemas import RawTransaction

logger = logging.getLogger("fraud_detection.feature_processor")

PROCESSED = Counter("fraud_features_processed_total", "Transactions turned into feature events.")
DEAD_LETTERS = Counter("fraud_features_dead_letters_total", "Messages sent to the DLQ.", ["stage"])
BATCH_SECONDS = Histogram(
    "fraud_features_batch_seconds",
    "Time to process one consumed batch (compute, publish, flush).",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)
INGEST_LAG = Histogram(
    "fraud_features_ingest_lag_seconds",
    "Wall-clock delay between publication by the producer and feature computation.",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

REDIS_ERRORS = (redis.ConnectionError, redis.TimeoutError)


class FeatureProcessor:
    def __init__(
        self,
        consumer: Consumer,
        producer: Producer,
        engineer: FeatureEngineer,
        settings: FeatureProcessorSettings,
        shutdown: GracefulShutdown,
    ) -> None:
        self.consumer = consumer
        self.producer = producer
        self.engineer = engineer
        self.settings = settings
        self.shutdown = shutdown
        self.tracker = DeliveryTracker()

    def run(self) -> None:
        self.consumer.subscribe([self.settings.topic_raw])
        logger.info("feature_processor_started", extra={"topic": self.settings.topic_raw})
        try:
            while not self.shutdown.requested:
                messages = self.consumer.consume(
                    num_messages=self.settings.batch_size, timeout=self.settings.poll_timeout_s
                )
                if messages:
                    self.process_batch(messages)
        finally:
            self.producer.flush(10)
            self.consumer.close()
            logger.info("feature_processor_stopped")

    def process_batch(self, messages: Sequence[Message]) -> None:
        started = time.perf_counter()
        valid: list[RawTransaction] = []
        for msg in messages:
            if not check_message_error(msg):
                continue
            try:
                valid.append(RawTransaction.model_validate_json(msg.value() or b""))
            except ValidationError as exc:
                self._dead_letter(msg, exc, "validate")

        if valid:
            events = retry_call(
                lambda: self.engineer.compute_batch(valid),
                retry_on=REDIS_ERRORS,
                shutdown=self.shutdown,
                description="feature_store",
                max_elapsed_s=120,
            )
            now = time.time()
            for event in events:
                self.producer.produce(
                    self.settings.topic_features,
                    key=event.user_id,
                    value=event.model_dump_json(),
                    on_delivery=self.tracker,
                )
                if event.emitted_at is not None:
                    INGEST_LAG.observe(max(0.0, now - event.emitted_at))

        self._commit(messages)
        PROCESSED.inc(len(valid))
        BATCH_SECONDS.observe(time.perf_counter() - started)

    def _dead_letter(self, msg: Message, error: Exception, stage: str) -> None:
        DEAD_LETTERS.labels(stage=stage).inc()
        logger.warning(
            "message_dead_lettered",
            extra={"stage": stage, "partition": msg.partition(), "offset": msg.offset()},
        )
        self.producer.produce(
            self.settings.topic_dlq,
            key=msg.key(),
            value=build_dead_letter(msg, error, stage),
            on_delivery=self.tracker,
        )

    def _commit(self, messages: Sequence[Message]) -> None:
        remaining = self.producer.flush(30)
        if remaining:
            raise DeliveryError(f"{remaining} messages still queued after flush")
        self.tracker.raise_for_failures()
        store_offsets(self.consumer, messages)


def main() -> None:
    settings = FeatureProcessorSettings()
    configure_logging("feature-processor", settings.log_level)
    start_metrics_server(settings.metrics_port, logger)
    shutdown = GracefulShutdown()

    admin = AdminClient({"bootstrap.servers": settings.kafka_bootstrap_servers})
    ensure_topics(
        admin,
        {
            settings.topic_raw: settings.topic_partitions,
            settings.topic_features: settings.topic_partitions,
            settings.topic_dlq: 1,
        },
        settings.topic_replication_factor,
    )
    engineer = FeatureEngineer(
        redis.Redis.from_url(settings.redis_url, health_check_interval=30),
        settings.feature_store_config(),
    )
    processor = FeatureProcessor(
        consumer=Consumer(
            consumer_config(settings.kafka_bootstrap_servers, settings.consumer_group)
        ),
        producer=Producer(producer_config(settings.kafka_bootstrap_servers)),
        engineer=engineer,
        settings=settings,
        shutdown=shutdown,
    )
    processor.run()


if __name__ == "__main__":
    main()
