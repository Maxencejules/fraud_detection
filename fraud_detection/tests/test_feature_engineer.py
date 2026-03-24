import sys
import os
import time
import math
import pytest
import fakeredis

# Add services/consumer to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "services", "consumer"))

from main import FeatureEngineer

def make_raw_tx(user_id="user_1", amount=100.0, is_fraud=False, merchant_id="m1"):
    return {
        "transaction_id": str(time.time()),
        "user_id": user_id,
        "merchant_id": merchant_id,
        "amount": amount,
        "currency": "USD",
        "timestamp": time.time(),
        "merchant_category": "retail",
        "country": "US",
        "card_present": True,
        "device_fingerprint": "abc",
        "_is_fraud": int(is_fraud)
    }

@pytest.fixture
def fe():
    r = fakeredis.FakeRedis(decode_responses=True)
    return FeatureEngineer(r)

def test_basic_features_present(fe):
    raw = make_raw_tx()
    out = fe.compute(raw)
    keys = [
        "amount", "amount_log", "amount_zscore", "tx_count_1h", "tx_count_24h",
        "tx_sum_1h", "tx_sum_24h", "unique_merchants_24h", "unique_countries_7d",
        "hour_of_day", "day_of_week", "is_weekend", "card_present",
        "merchant_fraud_rate_30d", "user_chargeback_rate", "label"
    ]
    for k in keys:
        assert k in out

def test_amount_log_is_log1p(fe):
    raw = make_raw_tx(amount=99.0)
    out = fe.compute(raw)
    assert out["amount_log"] == pytest.approx(math.log1p(99.0))

def test_velocity_increments(fe):
    user = "user_v"
    for _ in range(5):
        fe.compute(make_raw_tx(user_id=user))
    out = fe.compute(make_raw_tx(user_id=user))
    assert out["tx_count_1h"] == 6

def test_tx_sum_accumulates(fe):
    user = "user_s"
    for _ in range(4):
        fe.compute(make_raw_tx(user_id=user, amount=100.0))
    out = fe.compute(make_raw_tx(user_id=user, amount=100.0))
    assert out["tx_sum_1h"] == pytest.approx(500.0)

def test_unique_merchants_counted(fe):
    user = "user_m"
    merchants = ["m1", "m2", "m3", "m4"]
    for m in merchants:
        fe.compute(make_raw_tx(user_id=user, merchant_id=m))
    out = fe.compute(make_raw_tx(user_id=user, merchant_id="m5"))
    # HyperLogLog is approximate, check range
    assert 4 <= out["unique_merchants_24h"] <= 6

def test_label_propagated(fe):
    out_f = fe.compute(make_raw_tx(is_fraud=False))
    assert out_f["label"] == 0
    out_t = fe.compute(make_raw_tx(is_fraud=True))
    assert out_t["label"] == 1

def test_hour_of_day_in_range(fe):
    out = fe.compute(make_raw_tx())
    assert 0 <= out["hour_of_day"] <= 23

def test_amount_zscore_zero_for_first_tx(fe):
    # For a brand new user, mean defaults to amount, std defaults to 1.0
    # (amount - amount) / 1.0 = 0.0
    out = fe.compute(make_raw_tx(user_id="new_user", amount=50.0))
    assert out["amount_zscore"] == 0.0

def test_merchant_fraud_rate_from_redis(fe):
    m_id = "evil_m"
    fe.r.set(f"merchant:{m_id}:fraud_rate", "0.85")
    out = fe.compute(make_raw_tx(merchant_id=m_id))
    assert out["merchant_fraud_rate_30d"] == 0.85

def test_card_not_present_propagated(fe):
    raw = make_raw_tx()
    raw["card_present"] = False
    out = fe.compute(raw)
    assert out["card_present"] is False
