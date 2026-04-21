import pytest
import numpy as np
from unittest.mock import MagicMock, AsyncMock
from fastapi.testclient import TestClient

from services.predictor.main import app, state

@pytest.fixture
def client():
    # Lifespan is not automatically executed by TestClient unless using 'with'
    # But we patch state anyway
    state.redis = AsyncMock()
    state.model_version = "v1-test"
    with TestClient(app) as c:
        yield c

def make_mock_model(prob):
    model = MagicMock()
    # Predict returns array of probs
    model.predict.return_value = np.array([[1-prob, prob]])
    return model

def get_base_features():
    return {
        "transaction_id": "tx_123",
        "amount": 100.0,
        "amount_log": 4.61,
        "amount_zscore": 0.5,
        "tx_count_1h": 1,
        "tx_count_24h": 5,
        "tx_sum_1h": 100.0,
        "tx_sum_24h": 500.0,
        "unique_merchants_24h": 2,
        "unique_countries_7d": 1,
        "hour_of_day": 12,
        "day_of_week": 1,
        "is_weekend": False,
        "card_present": True,
        "merchant_fraud_rate_30d": 0.01,
        "user_chargeback_rate": 0.0
    }

def test_health(client):
    state.model = MagicMock()
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is True
    assert response.json()["model_version"] == "v1-test"

def test_approve_decision(client):
    state.model = make_mock_model(0.05)
    response = client.post("/predict", json=get_base_features())
    assert response.status_code == 200
    assert response.json()["decision"] == "APPROVE"
    assert response.json()["fraud_probability"] == 0.05

def test_review_decision(client):
    state.model = make_mock_model(0.55)
    response = client.post("/predict", json=get_base_features())
    assert response.status_code == 200
    assert response.json()["decision"] == "REVIEW"
    assert response.json()["fraud_probability"] == 0.55

def test_block_decision(client):
    state.model = make_mock_model(0.92)
    response = client.post("/predict", json=get_base_features())
    assert response.status_code == 200
    assert response.json()["decision"] == "BLOCK"
    assert response.json()["fraud_probability"] == 0.92

def test_model_not_loaded(client):
    state.model = None
    response = client.post("/predict", json=get_base_features())
    assert response.status_code == 503

def test_echo_transaction_id(client):
    state.model = make_mock_model(0.1)
    payload = get_base_features()
    payload["transaction_id"] = "unique_id_999"
    response = client.post("/predict", json=payload)
    assert response.json()["transaction_id"] == "unique_id_999"

def test_latency_header(client):
    state.model = make_mock_model(0.1)
    response = client.post("/predict", json=get_base_features())
    assert "X-Latency-Ms" in response.headers
    assert float(response.json()["latency_ms"]) > 0

def test_probability_range(client):
    state.model = make_mock_model(0.999)
    response = client.post("/predict", json=get_base_features())
    prob = response.json()["fraud_probability"]
    assert 0.0 <= prob <= 1.0
