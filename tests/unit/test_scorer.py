from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from typing import Any

import httpx
import pytest

from fraud_detection.config import ScorerSettings
from fraud_detection.kafka_utils import GracefulShutdown
from fraud_detection.schemas import (
    BatchPredictionResponse,
    Decision,
    DecisionEvent,
    FeatureEvent,
    FeatureVector,
    ScoredTransaction,
)
from fraud_detection.scorer import PredictorClient, PredictorUnavailableError, StreamScorer

SETTINGS = ScorerSettings()


def _feature_event(feature_payload: Any, tx_id: str, label: int | None = 0) -> FeatureEvent:
    return FeatureEvent(
        transaction_id=tx_id,
        user_id="u1",
        merchant_id="m1",
        event_time=1_767_225_600.0,
        emitted_at=1_767_225_600.0,
        processed_at=1_767_225_600.5,
        label=label,
        features=FeatureVector(**feature_payload()),
    )


class FakePredictor:
    def __init__(self, failures: int = 0, probability: float = 0.95) -> None:
        self.failures = failures
        self.probability = probability
        self.calls = 0

    def score(self, events: Sequence[FeatureEvent]) -> BatchPredictionResponse:
        self.calls += 1
        if self.failures:
            self.failures -= 1
            raise PredictorUnavailableError("model not loaded")
        return BatchPredictionResponse(
            items=[
                ScoredTransaction(
                    transaction_id=e.transaction_id,
                    fraud_probability=self.probability,
                    decision=Decision.BLOCK,
                )
                for e in events
            ],
            model_version="3",
            latency_ms=1.0,
        )

    def close(self) -> None:
        pass


def _scorer(
    kafka: Any, shutdown: GracefulShutdown, predictor: Any, batches: list[Any] | None = None
) -> tuple[StreamScorer, Any, Any]:
    """Returns the scorer plus its fake consumer and producer for assertions."""
    consumer, producer = kafka.Consumer(batches or [], shutdown), kafka.Producer()
    scorer = StreamScorer(
        consumer=consumer,
        producer=producer,
        predictor=predictor,
        settings=SETTINGS,
        shutdown=shutdown,
    )
    return scorer, consumer, producer


def test_decisions_carry_score_features_and_label(
    kafka: Any, shutdown: GracefulShutdown, feature_payload: Any
) -> None:
    events = [_feature_event(feature_payload, f"tx-{i}", label=i % 2) for i in range(3)]
    batch = [
        kafka.Message(e.model_dump_json().encode(), topic=SETTINGS.topic_features, offset=i)
        for i, e in enumerate(events)
    ]
    scorer, consumer, producer = _scorer(kafka, shutdown, FakePredictor())

    scorer.process_batch(batch)

    decisions = [
        DecisionEvent.model_validate_json(v) for v in producer.values(SETTINGS.topic_decisions)
    ]
    assert [d.transaction_id for d in decisions] == ["tx-0", "tx-1", "tx-2"]
    assert [d.label for d in decisions] == [0, 1, 0]
    assert all(d.decision == "BLOCK" and d.model_version == "3" for d in decisions)
    assert decisions[0].features == events[0].features
    assert [tp.offset for tp in consumer.stored] == [3]


def test_waits_for_the_predictor_instead_of_dropping_events(
    kafka: Any, shutdown: GracefulShutdown, feature_payload: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(GracefulShutdown, "wait", lambda *_: False)
    predictor = FakePredictor(failures=2)
    event = _feature_event(feature_payload, "tx-1")
    batch = [kafka.Message(event.model_dump_json().encode(), topic=SETTINGS.topic_features)]
    scorer, _consumer, producer = _scorer(kafka, shutdown, predictor)

    scorer.process_batch(batch)

    assert predictor.calls == 3
    assert len(producer.values(SETTINGS.topic_decisions)) == 1


def test_shutdown_during_outage_leaves_batch_uncommitted(
    kafka: Any, shutdown: GracefulShutdown, feature_payload: Any
) -> None:
    shutdown.request()
    event = _feature_event(feature_payload, "tx-1")
    batch = [kafka.Message(event.model_dump_json().encode(), topic=SETTINGS.topic_features)]
    scorer, consumer, _producer = _scorer(kafka, shutdown, FakePredictor(failures=5))

    with pytest.raises(PredictorUnavailableError):
        scorer.process_batch(batch)
    assert consumer.stored == []


def test_malformed_feature_events_are_dead_lettered(
    kafka: Any, shutdown: GracefulShutdown, feature_payload: Any
) -> None:
    good = _feature_event(feature_payload, "tx-1")
    batch = [
        kafka.Message(b"{}", topic=SETTINGS.topic_features, offset=0),
        kafka.Message(good.model_dump_json().encode(), topic=SETTINGS.topic_features, offset=1),
    ]
    scorer, _consumer, producer = _scorer(kafka, shutdown, FakePredictor())

    scorer.process_batch(batch)

    letters = [json.loads(v) for v in producer.values(SETTINGS.topic_dlq)]
    assert [(letter["stage"], letter["source_topic"]) for letter in letters] == [
        ("score", SETTINGS.topic_features)
    ]
    assert len(producer.values(SETTINGS.topic_decisions)) == 1


def test_mismatched_results_are_rejected(
    kafka: Any, shutdown: GracefulShutdown, feature_payload: Any
) -> None:
    class WrongIds(FakePredictor):
        def score(self, events: Sequence[FeatureEvent]) -> BatchPredictionResponse:
            response = super().score(events)
            response.items[0] = response.items[0].model_copy(update={"transaction_id": "other"})
            return response

    event = _feature_event(feature_payload, "tx-1")
    batch = [kafka.Message(event.model_dump_json().encode(), topic=SETTINGS.topic_features)]
    scorer, consumer, _producer = _scorer(kafka, shutdown, WrongIds())

    with pytest.raises(ValueError, match="does not match"):
        scorer.process_batch(batch)
    assert consumer.stored == []


def test_run_consumes_until_shutdown(
    kafka: Any, shutdown: GracefulShutdown, feature_payload: Any
) -> None:
    event = _feature_event(feature_payload, "tx-1")
    batches = [[kafka.Message(event.model_dump_json().encode(), topic=SETTINGS.topic_features)]]
    scorer, consumer, _producer = _scorer(kafka, shutdown, FakePredictor(), batches)

    scorer.run()

    assert consumer.subscribed == [SETTINGS.topic_features]
    assert consumer.closed is True


class TestPredictorClient:
    def _client(self, handler: Callable[[httpx.Request], httpx.Response]) -> PredictorClient:
        client = PredictorClient("http://predictor", timeout_s=1.0)
        client._client = httpx.Client(
            base_url="http://predictor", transport=httpx.MockTransport(handler)
        )
        return client

    def test_success(self, feature_payload: Any) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            assert request.url.path == "/v1/predict/batch"
            assert body["items"][0]["transaction_id"] == "tx-1"
            assert body["items"][0]["amount"] == 42.0
            return httpx.Response(
                200,
                json={
                    "items": [
                        {"transaction_id": "tx-1", "fraud_probability": 0.2, "decision": "REVIEW"}
                    ],
                    "model_version": "1",
                    "latency_ms": 0.5,
                },
            )

        response = self._client(handler).score([_feature_event(feature_payload, "tx-1")])

        assert response.items[0].decision == "REVIEW"

    @pytest.mark.parametrize("status", [500, 502, 503])
    def test_server_errors_are_retryable(self, status: int, feature_payload: Any) -> None:
        client = self._client(lambda _: httpx.Response(status, text="unavailable"))
        with pytest.raises(PredictorUnavailableError, match=str(status)):
            client.score([_feature_event(feature_payload, "tx-1")])

    def test_connection_errors_are_retryable(self, feature_payload: Any) -> None:
        def refuse(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused", request=request)

        with pytest.raises(PredictorUnavailableError, match="ConnectError"):
            self._client(refuse).score([_feature_event(feature_payload, "tx-1")])

    def test_contract_errors_fail_loudly(self, feature_payload: Any) -> None:
        client = self._client(lambda _: httpx.Response(422, json={"detail": "bad"}))
        with pytest.raises(httpx.HTTPStatusError):
            client.score([_feature_event(feature_payload, "tx-1")])
