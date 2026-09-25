"""Scores the feature stream through the predictor and publishes decisions.

Micro-batches of feature events are sent to ``POST /v1/predict/batch``; each result is
published to the decisions topic together with the features that produced it, which is
exactly what the monitor needs. While the predictor is unavailable (no model yet,
restarting) the scorer backs off and retries without committing, so the stream is
paused rather than dropped.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence

import httpx
from confluent_kafka import Consumer, Message, Producer
from confluent_kafka.admin import AdminClient
from prometheus_client import Counter, Histogram
from pydantic import ValidationError

from fraud_detection.config import ScorerSettings
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
from fraud_detection.schemas import BatchPredictionResponse, DecisionEvent, FeatureEvent

logger = logging.getLogger("fraud_detection.scorer")

DECISIONS = Counter("fraud_scorer_decisions_total", "Published decisions.", ["decision"])
DEAD_LETTERS = Counter("fraud_scorer_dead_letters_total", "Feature events sent to the DLQ.")
SCORING_SECONDS = Histogram(
    "fraud_scorer_request_seconds",
    "Round trip of one batch scoring request.",
    buckets=(0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)
END_TO_END_SECONDS = Histogram(
    "fraud_scorer_end_to_end_seconds",
    "Wall-clock time from publication by the producer to a published decision.",
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
)


class PredictorUnavailableError(RuntimeError):
    """The predictor could not score the batch right now (retryable)."""


class PredictorClient:
    def __init__(self, base_url: str, timeout_s: float) -> None:
        self._client = httpx.Client(base_url=base_url, timeout=timeout_s)

    def score(self, events: Sequence[FeatureEvent]) -> BatchPredictionResponse:
        payload = {
            "items": [
                {"transaction_id": e.transaction_id, **e.features.model_dump()} for e in events
            ]
        }
        started = time.perf_counter()
        try:
            response = self._client.post("/v1/predict/batch", json=payload)
        except httpx.TransportError as exc:
            raise PredictorUnavailableError(f"{type(exc).__name__}: {exc}") from exc
        SCORING_SECONDS.observe(time.perf_counter() - started)
        if response.status_code == 503 or response.status_code >= 500:
            raise PredictorUnavailableError(f"HTTP {response.status_code}: {response.text[:200]}")
        response.raise_for_status()  # other 4xx: a contract bug, fail loudly
        return BatchPredictionResponse.model_validate_json(response.content)

    def close(self) -> None:
        self._client.close()


class StreamScorer:
    def __init__(
        self,
        consumer: Consumer,
        producer: Producer,
        predictor: PredictorClient,
        settings: ScorerSettings,
        shutdown: GracefulShutdown,
    ) -> None:
        self.consumer = consumer
        self.producer = producer
        self.predictor = predictor
        self.settings = settings
        self.shutdown = shutdown
        self.tracker = DeliveryTracker()

    def run(self) -> None:
        self.consumer.subscribe([self.settings.topic_features])
        logger.info("scorer_started", extra={"topic": self.settings.topic_features})
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
            self.predictor.close()
            logger.info("scorer_stopped")

    def process_batch(self, messages: Sequence[Message]) -> None:
        events: list[FeatureEvent] = []
        for msg in messages:
            if not check_message_error(msg):
                continue
            try:
                events.append(FeatureEvent.model_validate_json(msg.value() or b""))
            except ValidationError as exc:
                DEAD_LETTERS.inc()
                self.producer.produce(
                    self.settings.topic_dlq,
                    key=msg.key(),
                    value=build_dead_letter(msg, exc, "score"),
                    on_delivery=self.tracker,
                )

        if events:
            response = retry_call(
                lambda: self.predictor.score(events),
                retry_on=(PredictorUnavailableError,),
                shutdown=self.shutdown,
                description="predictor",
                max_backoff_s=self.settings.max_backoff_s,
            )
            self._publish(events, response)

        remaining = self.producer.flush(30)
        if remaining:
            raise DeliveryError(f"{remaining} messages still queued after flush")
        self.tracker.raise_for_failures()
        store_offsets(self.consumer, messages)

    def _publish(self, events: Sequence[FeatureEvent], response: BatchPredictionResponse) -> None:
        if len(response.items) != len(events):
            raise ValueError(f"predictor returned {len(response.items)} results for {len(events)}")
        scored_at = time.time()
        for event, result in zip(events, response.items, strict=True):
            if result.transaction_id != event.transaction_id:
                raise ValueError(
                    f"result for {result.transaction_id} does not match {event.transaction_id}"
                )
            decision = DecisionEvent(
                transaction_id=event.transaction_id,
                user_id=event.user_id,
                merchant_id=event.merchant_id,
                event_time=event.event_time,
                emitted_at=event.emitted_at,
                scored_at=scored_at,
                fraud_probability=result.fraud_probability,
                decision=result.decision,
                model_version=response.model_version,
                label=event.label,
                features=event.features,
            )
            self.producer.produce(
                self.settings.topic_decisions,
                key=event.user_id,
                value=decision.model_dump_json(),
                on_delivery=self.tracker,
            )
            DECISIONS.labels(decision=result.decision.value).inc()
            if event.emitted_at is not None:
                END_TO_END_SECONDS.observe(max(0.0, scored_at - event.emitted_at))


def main() -> None:
    settings = ScorerSettings()
    configure_logging("scorer", settings.log_level)
    start_metrics_server(settings.metrics_port, logger)
    shutdown = GracefulShutdown()

    admin = AdminClient({"bootstrap.servers": settings.kafka_bootstrap_servers})
    ensure_topics(
        admin,
        {
            settings.topic_features: settings.topic_partitions,
            settings.topic_decisions: settings.topic_partitions,
            settings.topic_dlq: 1,
        },
        settings.topic_replication_factor,
    )
    scorer = StreamScorer(
        consumer=Consumer(
            consumer_config(settings.kafka_bootstrap_servers, settings.consumer_group)
        ),
        producer=Producer(producer_config(settings.kafka_bootstrap_servers)),
        predictor=PredictorClient(settings.predictor_url, settings.predictor_timeout_s),
        settings=settings,
        shutdown=shutdown,
    )
    scorer.run()


if __name__ == "__main__":
    main()
