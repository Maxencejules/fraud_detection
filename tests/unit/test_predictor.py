from __future__ import annotations

import asyncio
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from fraud_detection.config import PredictorSettings
from fraud_detection.features import FEATURE_COLUMNS
from fraud_detection.modeling import EnsembleScorer
from fraud_detection.predictor import ModelManager, create_app
from fraud_detection.registry import LoadedModel, ModelRef


class FakeRegistry:
    """Stands in for MLflow: an alias pointer plus loadable versions."""

    def __init__(self, scorer: EnsembleScorer) -> None:
        self.scorer = scorer
        self.version: str | None = "1"
        self.fail_load = False
        self.feature_columns = FEATURE_COLUMNS
        self.loads = 0

    def resolve(self) -> ModelRef | None:
        if self.version is None:
            return None
        return ModelRef(
            "fraud-detector", self.version, f"run-{self.version}", f"models:/fd/{self.version}"
        )

    def load(self, uri: str, version: str | None) -> LoadedModel:
        self.loads += 1
        if self.fail_load:
            raise RuntimeError("artifact store unavailable")
        scorer = self.scorer
        if self.feature_columns != FEATURE_COLUMNS:
            scorer = EnsembleScorer(
                scorer.xgb_booster, scorer.lgb_booster, scorer.calibrator, self.feature_columns
            )
        return LoadedModel(scorer, uri, version or uri, f"run-{version}", loaded_at=0.0)


@pytest.fixture
def registry(tiny_scorer: EnsembleScorer) -> FakeRegistry:
    return FakeRegistry(tiny_scorer)


def _client(registry: FakeRegistry, **settings: Any) -> TestClient:
    manager = ModelManager(resolver=registry.resolve, loader=registry.load, poll_interval_s=0)
    app = create_app(PredictorSettings(**settings), manager)
    return TestClient(app)


@pytest.fixture
def client(registry: FakeRegistry) -> Iterator[TestClient]:
    with _client(registry, admin_token="s3cret-admin-token") as test_client:
        yield test_client


def test_health_is_always_ok(registry: FakeRegistry) -> None:
    registry.version = None
    with _client(registry) as client:
        assert client.get("/health").json() == {"status": "ok"}


def test_ready_reports_missing_model(registry: FakeRegistry) -> None:
    registry.version = None
    with _client(registry) as client:
        response = client.get("/ready")
    assert response.status_code == 503
    assert "no model registered" in response.json()["reason"]


def test_ready_when_model_loaded(client: TestClient) -> None:
    assert client.get("/ready").json() == {"status": "ready", "model_version": "1"}


def test_predict_returns_calibrated_score_and_decision(
    client: TestClient, feature_payload: Any
) -> None:
    response = client.post("/v1/predict", json={"transaction_id": "tx-1", **feature_payload()})

    assert response.status_code == 200
    body = response.json()
    assert body["transaction_id"] == "tx-1"
    assert 0.0 <= body["fraud_probability"] <= 1.0
    thresholds = PredictorSettings().thresholds()
    assert body["decision"] == thresholds.decide(body["fraud_probability"]).value
    assert body["model_version"] == "1"
    assert float(response.headers["X-Latency-Ms"]) >= 0


def test_high_risk_payload_scores_higher(client: TestClient, feature_payload: Any) -> None:
    def score(zscore: float) -> float:
        payload = {"transaction_id": "tx", **feature_payload(amount_zscore=zscore)}
        return float(client.post("/v1/predict", json=payload).json()["fraud_probability"])

    assert score(0.99) > score(0.0)


@pytest.mark.parametrize(
    ("override", "field"),
    [
        ({"hour_of_day": 24}, "hour_of_day"),
        ({"amount": -1.0}, "amount"),
        ({"merchant_fraud_rate_30d": 1.5}, "merchant_fraud_rate_30d"),
        ({"unexpected": 1}, "unexpected"),
        ({"transaction_id": "has spaces"}, "transaction_id"),
    ],
)
def test_predict_validates_input(
    client: TestClient, feature_payload: Any, override: dict[str, Any], field: str
) -> None:
    payload = {"transaction_id": "tx-1", **feature_payload(), **override}

    response = client.post("/v1/predict", json=payload)

    assert response.status_code == 422
    assert field in response.text


def test_missing_feature_is_rejected(client: TestClient, feature_payload: Any) -> None:
    payload = {"transaction_id": "tx-1", **feature_payload()}
    del payload["card_present"]

    assert client.post("/v1/predict", json=payload).status_code == 422


def test_batch_preserves_order(client: TestClient, feature_payload: Any) -> None:
    items = [{"transaction_id": f"tx-{i}", **feature_payload(amount=10.0 + i)} for i in range(5)]

    response = client.post("/v1/predict/batch", json={"items": items})

    assert response.status_code == 200
    assert [i["transaction_id"] for i in response.json()["items"]] == [f"tx-{i}" for i in range(5)]
    assert response.json()["model_version"] == "1"


def test_batch_limit_is_enforced(registry: FakeRegistry, feature_payload: Any) -> None:
    items = [{"transaction_id": f"tx-{i}", **feature_payload()} for i in range(3)]
    with _client(registry, max_batch_size=2) as client:
        response = client.post("/v1/predict/batch", json={"items": items})
    assert response.status_code == 413


def test_empty_batch_is_rejected(client: TestClient) -> None:
    assert client.post("/v1/predict/batch", json={"items": []}).status_code == 422


def test_scoring_without_model_returns_503(registry: FakeRegistry, feature_payload: Any) -> None:
    registry.version = None
    with _client(registry) as client:
        response = client.post("/v1/predict", json={"transaction_id": "tx", **feature_payload()})
    assert response.status_code == 503


def test_model_info(client: TestClient) -> None:
    body = client.get("/v1/model").json()

    assert body["version"] == "1"
    assert body["feature_columns"] == list(FEATURE_COLUMNS)
    assert body["thresholds"] == {"review": 0.1, "block": 0.9}


def test_metrics_expose_predictions(client: TestClient, feature_payload: Any) -> None:
    client.post("/v1/predict", json={"transaction_id": "tx", **feature_payload()})

    text = client.get("/metrics").text

    assert "fraud_predictor_predictions_total" in text
    assert 'fraud_predictor_model_info{run_id="run-1",version="1"} 1.0' in text


def test_request_id_is_echoed_or_generated(client: TestClient) -> None:
    assert client.get("/health", headers={"X-Request-ID": "abc"}).headers["X-Request-ID"] == "abc"
    assert len(client.get("/health").headers["X-Request-ID"]) == 32


def test_internal_errors_do_not_leak_details(
    registry: FakeRegistry, feature_payload: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    def explode(_: Any) -> Any:
        raise RuntimeError("secret internal detail")

    with _client(registry) as client:
        monkeypatch.setattr(registry.scorer, "predict_proba", explode)
        response = client.post("/v1/predict", json={"transaction_id": "tx", **feature_payload()})

    assert response.status_code == 500
    assert "secret" not in response.text
    assert response.json()["request_id"]


class TestAdminReload:
    def test_disabled_without_token(self, registry: FakeRegistry) -> None:
        with _client(registry) as client:
            assert client.post("/v1/admin/reload").status_code == 403

    def test_rejects_wrong_token(self, client: TestClient) -> None:
        response = client.post("/v1/admin/reload", headers={"Authorization": "Bearer nope"})
        assert response.status_code == 401

    def test_rejects_non_ascii_token(self, client: TestClient) -> None:
        response = client.post(
            "/v1/admin/reload", headers={b"Authorization": "Bearer café".encode()}
        )
        assert response.status_code == 401

    def test_reports_a_failed_reload(self, client: TestClient, registry: FakeRegistry) -> None:
        registry.version, registry.fail_load = "2", True

        response = client.post(
            "/v1/admin/reload", headers={"Authorization": "Bearer s3cret-admin-token"}
        )

        assert response.status_code == 200
        assert response.json()["reloaded"] is False
        assert response.json()["model_version"] == "1"
        assert "artifact store unavailable" in response.json()["error"]
        ready = client.get("/ready")
        assert ready.status_code == 200  # still serving version 1
        assert "artifact store unavailable" in ready.json()["last_reload_error"]

    def test_reloads_with_valid_token(self, client: TestClient, registry: FakeRegistry) -> None:
        registry.version = "2"
        response = client.post(
            "/v1/admin/reload", headers={"Authorization": "Bearer s3cret-admin-token"}
        )

        assert response.status_code == 200
        assert response.json() == {"reloaded": True, "model_version": "2"}


class TestModelManager:
    def _manager(self, registry: FakeRegistry) -> ModelManager:
        return ModelManager(resolver=registry.resolve, loader=registry.load, poll_interval_s=0)

    def test_swaps_when_alias_moves(self, registry: FakeRegistry) -> None:
        manager = self._manager(registry)
        asyncio.run(manager.refresh())
        registry.version = "2"

        assert asyncio.run(manager.refresh()) is True
        assert manager.current is not None
        assert manager.current.version == "2"

    def test_does_not_reload_unchanged_version(self, registry: FakeRegistry) -> None:
        manager = self._manager(registry)
        asyncio.run(manager.refresh())

        assert asyncio.run(manager.refresh()) is False
        assert registry.loads == 1

    def test_keeps_serving_when_new_version_fails_to_load(self, registry: FakeRegistry) -> None:
        manager = self._manager(registry)
        asyncio.run(manager.refresh())
        registry.version, registry.fail_load = "2", True

        assert asyncio.run(manager.refresh()) is False
        assert manager.current is not None
        assert manager.current.version == "1"
        assert manager.last_error is not None
        assert "artifact store unavailable" in manager.last_error

    def test_rejects_model_with_different_feature_contract(self, registry: FakeRegistry) -> None:
        registry.feature_columns = tuple(reversed(FEATURE_COLUMNS))
        manager = self._manager(registry)

        assert asyncio.run(manager.refresh()) is False
        assert manager.current is None
        assert manager.last_error is not None
        assert "do not match" in manager.last_error

    def test_background_polling_picks_up_new_versions(self, registry: FakeRegistry) -> None:
        async def scenario() -> str | None:
            manager = ModelManager(
                resolver=registry.resolve, loader=registry.load, poll_interval_s=0.01
            )
            await manager.start()
            registry.version = "7"
            await asyncio.sleep(0.1)
            await manager.stop()
            return manager.current.version if manager.current else None

        assert asyncio.run(scenario()) == "7"


def test_cors_is_opt_in(registry: FakeRegistry) -> None:
    origin = {"Origin": "https://example.com"}
    with _client(registry) as client:
        assert "access-control-allow-origin" not in client.get("/health", headers=origin).headers
    with _client(registry, cors_allow_origins="https://example.com") as client:
        allowed = client.get("/health", headers=origin).headers
    assert allowed["access-control-allow-origin"] == "https://example.com"
