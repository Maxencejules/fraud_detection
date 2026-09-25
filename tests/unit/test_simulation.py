from __future__ import annotations

from collections import defaultdict
from itertools import pairwise

import numpy as np
import pytest

from fraud_detection.simulation import (
    DAY,
    HOME_COUNTRIES,
    Population,
    SimulationConfig,
    TransactionSimulator,
)

T0 = 1_767_225_600.0
SMALL = SimulationConfig(seed=7, n_users=400, n_merchants=120)


@pytest.fixture(scope="module")
def two_weeks() -> list:  # type: ignore[type-arg]
    return list(TransactionSimulator(SMALL, stream="history").events(T0, T0 + 14 * DAY))


def test_population_is_deterministic() -> None:
    first, second = Population.from_config(SMALL), Population.from_config(SMALL)

    assert first.user_ids == second.user_ids
    assert first.user_country == second.user_country
    np.testing.assert_array_equal(first.user_daily_rate, second.user_daily_rate)
    pairs = zip(first.user_favourites, second.user_favourites, strict=True)
    assert all(np.array_equal(a, b) for a, b in pairs)


def test_population_home_countries_are_known() -> None:
    population = Population.from_config(SMALL)

    assert set(population.user_country) <= {c for c, _ in HOME_COUNTRIES}
    assert population.high_risk_merchants.size > 0


def test_stream_is_reproducible_per_stream_name() -> None:
    def ids(stream: str) -> list[str]:
        sim = TransactionSimulator(SMALL, stream=stream)
        return [tx.transaction_id for tx in sim.events(T0, T0 + DAY)]

    assert ids("history") == ids("history")
    assert ids("history") != ids("live")


def test_events_are_time_ordered_and_bounded(two_weeks: list) -> None:  # type: ignore[type-arg]
    timestamps = [tx.timestamp for tx in two_weeks]

    assert timestamps == sorted(timestamps)
    assert timestamps[0] >= T0
    assert timestamps[-1] < T0 + 14 * DAY


def test_volume_matches_configured_rate(two_weeks: list) -> None:  # type: ignore[type-arg]
    per_user_day = len(two_weeks) / SMALL.n_users / 14

    assert per_user_day == pytest.approx(SMALL.mean_tx_per_user_per_day, rel=0.15)


def test_fraud_is_rare_but_present(two_weeks: list) -> None:  # type: ignore[type-arg]
    fraud_rate = sum(bool(tx.is_fraud) for tx in two_weeks) / len(two_weeks)

    assert 0.004 < fraud_rate < 0.03


def test_fraud_is_bursty_and_card_not_present(two_weeks: list) -> None:  # type: ignore[type-arg]
    frauds = [tx for tx in two_weeks if tx.is_fraud]
    by_user: dict[str, list[float]] = defaultdict(list)
    for tx in frauds:
        by_user[tx.user_id].append(tx.timestamp)
    gaps = [b - a for times in by_user.values() for a, b in pairwise(times)]

    assert np.median(gaps) < 3_600
    assert np.mean([tx.card_present for tx in frauds]) < 0.2


def test_legitimate_activity_follows_diurnal_profile(two_weeks: list) -> None:  # type: ignore[type-arg]
    hours = np.array([int((tx.timestamp % DAY) // 3_600) for tx in two_weeks if not tx.is_fraud])

    night = np.isin(hours, [2, 3, 4]).mean()
    midday = np.isin(hours, [11, 12, 13]).mean()
    assert midday > 5 * night


def test_live_stream_can_run_unbounded() -> None:
    stream = TransactionSimulator(SMALL, stream="live").events(T0)

    first = [next(stream) for _ in range(50)]

    assert len({tx.transaction_id for tx in first}) == 50
