from __future__ import annotations

import math
from typing import Any

import fakeredis
import pytest

from fraud_detection.features import (
    DAY,
    FEATURE_COLUMNS,
    HOUR,
    USER_HISTORY_SECONDS,
    FeatureEngineer,
    FeatureStoreConfig,
    build_feature_vector,
    encode_member,
    merchant_window_days,
    utc_day,
)
from fraud_detection.schemas import FeatureVector, RawTransaction

T0 = 1_767_225_600.0  # 2026-01-01T00:00:00Z, a Thursday
CONFIG = FeatureStoreConfig()


def _snapshot(client: fakeredis.FakeRedis) -> dict[Any, Any]:
    state: dict[Any, Any] = {}
    for key in sorted(client.keys("*")):
        if client.type(key) == b"zset":
            state[key] = client.zrange(key, 0, -1, withscores=True)
        else:  # HyperLogLog (fakeredis models it as a set)
            state[key] = client.pfcount(key)
    return state


def test_feature_columns_follow_schema_order() -> None:
    assert tuple(FeatureVector.model_fields) == FEATURE_COLUMNS
    assert len(FEATURE_COLUMNS) == 17
    assert "label" not in FEATURE_COLUMNS


class TestBuildFeatureVector:
    def test_first_transaction_has_neutral_history_features(self, make_tx: Any) -> None:
        tx = make_tx(amount=42.5)
        vector = build_feature_vector(tx, [encode_member(tx)], 0, 0, 0, CONFIG)

        assert vector.tx_count_1h == 1
        assert vector.tx_sum_24h == pytest.approx(42.5)
        assert vector.amount_log == pytest.approx(math.log1p(42.5))
        assert vector.amount_zscore == 0.0
        assert vector.seconds_since_last_tx == USER_HISTORY_SECONDS
        assert vector.is_new_country is False
        assert vector.user_chargeback_rate == 0.0
        # With no merchant history the smoothed rate equals the prior.
        assert vector.merchant_fraud_rate_30d == pytest.approx(CONFIG.merchant_prior_rate)

    def test_windows_are_relative_to_event_time(self, make_tx: Any) -> None:
        history = [
            make_tx(amount=10.0, timestamp=T0 - 30 * 60, merchant_id="m1", country="US"),
            make_tx(amount=20.0, timestamp=T0 - 2 * HOUR, merchant_id="m2", country="US"),
            make_tx(amount=40.0, timestamp=T0 - 25 * HOUR, merchant_id="m3", country="CA"),
            make_tx(amount=80.0, timestamp=T0 - 8 * DAY, merchant_id="m4", country="FR"),
        ]
        current = make_tx(amount=5.0, timestamp=T0, merchant_id="m1", country="US")
        members = [encode_member(t) for t in [*history, current]]

        vector = build_feature_vector(current, members, 0, 0, 0, CONFIG)

        assert vector.tx_count_1h == 2
        assert vector.tx_sum_1h == pytest.approx(15.0)
        assert vector.tx_count_24h == 3
        assert vector.tx_sum_24h == pytest.approx(35.0)
        assert vector.unique_merchants_24h == 2  # m1 (twice) and m2
        assert vector.unique_countries_7d == 2  # US and CA; FR is 8 days old
        assert vector.seconds_since_last_tx == pytest.approx(30 * 60)

    def test_amount_zscore_uses_prior_transactions_only(self, make_tx: Any) -> None:
        prior = [make_tx(amount=a, timestamp=T0 - (i + 1) * HOUR) for i, a in enumerate([10, 20])]
        current = make_tx(amount=45.0)
        members = [encode_member(t) for t in [*prior, current]]

        vector = build_feature_vector(current, members, 0, 0, 0, CONFIG)

        assert vector.amount_zscore == pytest.approx((45.0 - 15.0) / 5.0)

    def test_amount_zscore_is_bounded(self, make_tx: Any) -> None:
        prior = [make_tx(amount=1.0, timestamp=T0 - (i + 1) * HOUR) for i in range(5)]
        current = make_tx(amount=5_000.0)
        members = [encode_member(t) for t in [*prior, current]]

        vector = build_feature_vector(current, members, 0, 0, 0, CONFIG)

        assert vector.amount_zscore == 50.0

    def test_new_country_is_relative_to_user_history(self, make_tx: Any) -> None:
        prior = make_tx(country="US", timestamp=T0 - DAY)
        current = make_tx(country="BR")
        members = [encode_member(prior), encode_member(current)]

        assert build_feature_vector(current, members, 0, 0, 0, CONFIG).is_new_country is True

    def test_later_events_are_ignored(self, make_tx: Any) -> None:
        later = make_tx(amount=999.0, timestamp=T0 + 60)
        current = make_tx(amount=10.0)
        members = [encode_member(later), encode_member(current)]

        vector = build_feature_vector(current, members, 0, 0, 0, CONFIG)

        assert vector.tx_count_1h == 1
        assert vector.tx_sum_1h == pytest.approx(10.0)

    def test_chargeback_rate_only_counts_labelled_period(self, make_tx: Any) -> None:
        recent = make_tx(timestamp=T0 - HOUR)  # label not known yet
        labelled = [make_tx(timestamp=T0 - CONFIG.label_delay_seconds - i * DAY) for i in (1, 2)]
        current = make_tx()
        members = [encode_member(t) for t in [recent, *labelled, current]]

        vector = build_feature_vector(current, members, 1, 0, 0, CONFIG)

        assert vector.user_chargeback_rate == pytest.approx(0.5)

    def test_merchant_rate_is_smoothed_towards_prior(self, make_tx: Any) -> None:
        tx = make_tx()

        vector = build_feature_vector(tx, [encode_member(tx)], 0, 150, 30, CONFIG)

        expected = (30 + CONFIG.merchant_prior_weight * CONFIG.merchant_prior_rate) / (
            150 + CONFIG.merchant_prior_weight
        )
        assert vector.merchant_fraud_rate_30d == pytest.approx(expected)

    def test_calendar_features_use_utc(self, make_tx: Any) -> None:
        saturday_noon = T0 + 2 * DAY + 12 * HOUR
        tx = make_tx(timestamp=saturday_noon)

        vector = build_feature_vector(tx, [encode_member(tx)], 0, 0, 0, CONFIG)

        assert (vector.hour_of_day, vector.day_of_week, vector.is_weekend) == (12, 5, True)


class TestFeatureEngineer:
    def test_reprocessing_is_idempotent(
        self, engineer: FeatureEngineer, redis_client: fakeredis.FakeRedis, make_tx: Any
    ) -> None:
        txs = [make_tx(timestamp=T0 + i * 60, is_fraud=i == 1) for i in range(3)]
        first = engineer.compute_batch(txs)
        state = _snapshot(redis_client)

        replay = engineer.compute_batch(txs)

        assert [e.features for e in replay] == [e.features for e in first]
        assert _snapshot(redis_client) == state

    def test_batch_matches_sequential_processing(self, make_tx: Any) -> None:
        txs = [
            make_tx(user_id=f"u{i % 2}", timestamp=T0 + i * 90, amount=10.0 + i) for i in range(6)
        ]
        batched = FeatureEngineer(fakeredis.FakeRedis()).compute_batch(txs)
        sequential_engineer = FeatureEngineer(fakeredis.FakeRedis())
        sequential = [sequential_engineer.compute(tx) for tx in txs]

        assert [e.features for e in batched] == [e.features for e in sequential]

    def test_fraud_burst_cannot_see_its_own_labels(
        self, engineer: FeatureEngineer, make_tx: Any
    ) -> None:
        burst = [make_tx(timestamp=T0 + i * 120, is_fraud=True) for i in range(5)]
        events = engineer.compute_batch(burst)
        assert all(e.features.user_chargeback_rate == 0.0 for e in events)

        after_delay = make_tx(timestamp=T0 + CONFIG.label_delay_seconds + DAY)
        assert engineer.compute(after_delay).features.user_chargeback_rate == pytest.approx(1.0)

    def test_merchant_fraud_becomes_visible_after_label_delay(
        self, engineer: FeatureEngineer, make_tx: Any
    ) -> None:
        engineer.compute(make_tx(merchant_id="m-risky", is_fraud=True, user_id="victim"))
        same_day = engineer.compute(make_tx(merchant_id="m-risky", timestamp=T0 + HOUR))
        later = engineer.compute(
            make_tx(merchant_id="m-risky", timestamp=T0 + CONFIG.label_delay_seconds + 2 * DAY)
        )

        prior = CONFIG.merchant_prior_rate
        assert same_day.features.merchant_fraud_rate_30d == pytest.approx(prior)
        assert later.features.merchant_fraud_rate_30d > prior

    def test_state_older_than_retention_is_trimmed(
        self, engineer: FeatureEngineer, redis_client: fakeredis.FakeRedis, make_tx: Any
    ) -> None:
        engineer.compute(make_tx(timestamp=T0))
        engineer.compute(make_tx(timestamp=T0 + CONFIG.user_retention_seconds + DAY))

        assert redis_client.zcard(CONFIG.user_key("u000001", "tx")) == 1

    def test_emits_identifiers_and_label(self, engineer: FeatureEngineer, make_tx: Any) -> None:
        tx = make_tx(is_fraud=True, emitted_at=T0 + 1)

        event = engineer.compute(tx)

        assert (event.transaction_id, event.user_id, event.merchant_id) == (
            tx.transaction_id,
            tx.user_id,
            tx.merchant_id,
        )
        assert event.label == 1
        assert event.event_time == T0
        assert event.emitted_at == T0 + 1

    def test_unknown_label_is_propagated_as_none(
        self, engineer: FeatureEngineer, make_tx: Any
    ) -> None:
        assert engineer.compute(make_tx(is_fraud=None)).label is None

    def test_decoded_and_raw_clients_agree(self, make_tx: Any) -> None:
        txs = [make_tx(timestamp=T0 + i * 60) for i in range(3)]
        raw = FeatureEngineer(fakeredis.FakeRedis()).compute_batch(txs)
        decoded = FeatureEngineer(fakeredis.FakeRedis(decode_responses=True)).compute_batch(txs)

        assert [e.features for e in raw] == [e.features for e in decoded]

    def test_empty_batch(self, engineer: FeatureEngineer) -> None:
        assert engineer.compute_batch([]) == []


def test_keys_use_cluster_hash_tags() -> None:
    assert CONFIG.user_key("u1", "tx") == "fd:u:{u1}:tx"
    assert CONFIG.merchant_key("m1", "fraud", 3) == "fd:m:{m1}:fraud:3"


def test_merchant_window_excludes_days_without_known_labels() -> None:
    days = merchant_window_days(T0, CONFIG)

    assert len(days) == CONFIG.merchant_window_days
    assert days[-1] == utc_day(T0 - CONFIG.label_delay_seconds) - 1


def test_merchant_counters_outside_event_time_retention_are_deleted(
    engineer: FeatureEngineer, redis_client: fakeredis.FakeRedis, make_tx: Any
) -> None:
    first_day = utc_day(T0)
    for offset in range(80):  # one purchase a day at the same merchant
        engineer.compute(
            make_tx(transaction_id=f"t{offset}", merchant_id="m1", timestamp=T0 + offset * DAY)
        )

    kept = [
        offset
        for offset in range(80)
        if redis_client.exists(CONFIG.merchant_key("m1", "tx", first_day + offset))
    ]
    assert kept == list(range(80 - CONFIG.merchant_retention_days, 80))
    readable = merchant_window_days(T0 + 79 * DAY, CONFIG)  # what features can still read
    assert {day - first_day for day in readable} <= set(kept)


def test_raw_transaction_rejects_malformed_identifiers() -> None:
    with pytest.raises(ValueError, match="transaction_id"):
        RawTransaction(
            transaction_id="bad|id",
            user_id="u",
            merchant_id="m",
            merchant_category="retail",
            amount=1.0,
            country="US",
            card_present=True,
            timestamp=T0,
        )
