"""Synthetic card-transaction stream with realistic, overlapping fraud patterns.

The same simulator feeds both the live Kafka producer and the historical backfill used
to build the training set, so the model is trained on exactly the behaviour it later
scores.

Behaviour model
---------------
* **Legitimate activity** is a superposition of per-user Poisson processes with a
  diurnal intensity profile. Each user has a home country, a typical spend
  distribution, favourite merchants and a card-present propensity. Legitimate noise
  (travel, occasional large purchases, online shopping) deliberately overlaps with
  fraud so the problem is not trivially separable.
* **Account takeover (ATO)** incidents start bursts of card-not-present transactions
  minutes apart: optional low-value card testing followed by high-value purchases,
  often from a country new to the user and at high-risk merchants. A small share of
  users is re-victimised far more often (leaked credentials).
* **Opportunistic fraud** replaces a small share of ordinary transactions with a single
  high-value card-not-present purchase at a high-risk merchant.

Fraud is defined by behaviour relative to the user's own history, never by country of
origin: the fraudster's country is drawn uniformly from countries other than the
victim's home country.

The simulator runs in *event time*. The live producer paces emission in wall-clock
time, which implies a simulated-time speed-up; features are computed from event time,
so online and offline semantics stay identical.
"""

from __future__ import annotations

import heapq
import math
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np

from fraud_detection.schemas import RawTransaction

DAY = 86_400.0

HOME_COUNTRIES: tuple[tuple[str, float], ...] = (
    ("US", 0.60),
    ("CA", 0.10),
    ("GB", 0.09),
    ("DE", 0.07),
    ("FR", 0.06),
    ("ES", 0.04),
    ("IT", 0.04),
)
# Destinations for travel and for fraudsters' locations. Drawn uniformly (excluding the
# victim's home country) so no single country is associated with fraud.
ALL_COUNTRIES: tuple[str, ...] = (
    "US", "CA", "GB", "DE", "FR", "ES", "IT", "NL", "BE", "PT", "IE", "SE", "PL", "MX",
    "BR", "AR", "JP", "KR", "AU", "NZ", "IN", "SG", "ZA", "AE", "TR",
)  # fmt: skip

CATEGORIES: tuple[str, ...] = (
    "grocery",
    "restaurants",
    "fuel",
    "retail",
    "travel",
    "entertainment",
    "electronics",
    "digital_goods",
    "health",
    "utilities",
)
CATEGORY_WEIGHTS: tuple[float, ...] = (0.2, 0.18, 0.1, 0.17, 0.05, 0.08, 0.06, 0.06, 0.06, 0.04)
HIGH_RISK_CATEGORIES: frozenset[str] = frozenset({"electronics", "digital_goods", "travel"})
ONLINE_CATEGORIES: frozenset[str] = frozenset({"digital_goods", "utilities"})

# Relative transaction intensity by UTC hour (legitimate activity only).
DIURNAL_PROFILE: tuple[float, ...] = (
    0.25, 0.15, 0.1, 0.08, 0.08, 0.12, 0.3, 0.55, 0.8, 0.95, 1.0, 1.0,
    1.0, 1.0, 0.95, 0.95, 1.0, 1.0, 1.0, 0.95, 0.85, 0.7, 0.5, 0.35,
)  # fmt: skip


@dataclass(frozen=True)
class SimulationConfig:
    seed: int = 42
    n_users: int = 3_000
    n_merchants: int = 600
    mean_tx_per_user_per_day: float = 1.5
    risky_user_share: float = 0.05
    ato_rate_per_user_day: float = 0.0012
    risky_user_ato_multiplier: float = 10.0
    high_risk_merchant_share: float = 0.08
    opportunistic_fraud_prob: float = 0.003
    travel_prob: float = 0.03
    large_purchase_prob: float = 0.02


@dataclass(frozen=True)
class Population:
    """Users and merchants. Fully determined by ``SimulationConfig.seed``."""

    user_ids: tuple[str, ...]
    user_country: tuple[str, ...]
    user_log_mean: np.ndarray
    user_log_sigma: np.ndarray
    user_daily_rate: np.ndarray
    user_card_present_p: np.ndarray
    user_risky: np.ndarray
    user_device: tuple[str, ...]
    user_favourites: tuple[np.ndarray, ...]
    merchant_ids: tuple[str, ...]
    merchant_category: tuple[str, ...]
    merchant_country: tuple[str, ...]
    merchant_high_risk: np.ndarray
    merchants_by_country: dict[str, np.ndarray] = field(repr=False)
    high_risk_merchants: np.ndarray = field(repr=False)

    @classmethod
    def from_config(cls, config: SimulationConfig) -> Population:
        rng = np.random.default_rng([config.seed, 0])
        countries = [c for c, _ in HOME_COUNTRIES]
        country_p = np.array([w for _, w in HOME_COUNTRIES])
        country_p /= country_p.sum()

        n_m = config.n_merchants
        merchant_category = tuple(
            str(c) for c in rng.choice(CATEGORIES, size=n_m, p=np.array(CATEGORY_WEIGHTS))
        )
        merchant_country = tuple(str(c) for c in rng.choice(countries, size=n_m, p=country_p))
        risk_boost = np.array(
            [3.0 if c in HIGH_RISK_CATEGORIES else 0.5 for c in merchant_category]
        )
        risk_p = np.clip(config.high_risk_merchant_share * risk_boost, 0.0, 0.9)
        merchant_high_risk = rng.random(n_m) < risk_p
        merchants_by_country = {
            c: np.flatnonzero(np.array(merchant_country) == c) for c in countries
        }
        # Guarantee every home country has at least one merchant.
        for c in countries:
            if merchants_by_country[c].size == 0:
                merchants_by_country[c] = np.arange(n_m)
        high_risk = np.flatnonzero(merchant_high_risk)
        if high_risk.size == 0:
            high_risk = np.arange(n_m)

        n_u = config.n_users
        user_country = tuple(str(c) for c in rng.choice(countries, size=n_u, p=country_p))
        favourites = []
        for c in user_country:
            local = merchants_by_country[c]
            k = min(int(local.size), int(rng.integers(4, 12)))
            favourites.append(rng.choice(local, size=k, replace=False))

        return cls(
            user_ids=tuple(f"u{i:06d}" for i in range(n_u)),
            user_country=user_country,
            user_log_mean=rng.normal(3.3, 0.55, n_u),
            user_log_sigma=rng.uniform(0.5, 1.0, n_u),
            user_daily_rate=rng.gamma(2.0, config.mean_tx_per_user_per_day / 2.0, n_u),
            user_card_present_p=rng.beta(6.0, 3.0, n_u),
            user_risky=rng.random(n_u) < config.risky_user_share,
            user_device=tuple(f"d{rng.integers(0, 2**48):012x}" for _ in range(n_u)),
            user_favourites=tuple(favourites),
            merchant_ids=tuple(f"m{j:05d}" for j in range(n_m)),
            merchant_category=merchant_category,
            merchant_country=merchant_country,
            merchant_high_risk=merchant_high_risk,
            merchants_by_country=merchants_by_country,
            high_risk_merchants=high_risk,
        )


class TransactionSimulator:
    """Generates an event-time ordered stream of transactions."""

    def __init__(
        self,
        config: SimulationConfig,
        population: Population | None = None,
        stream: str | int = "history",
    ):
        self.config = config
        self.population = population or Population.from_config(config)
        stream_key = stream if isinstance(stream, int) else _stable_hash(stream)
        self._rng = np.random.default_rng([config.seed, 1, stream_key])

        pop = self.population
        rates = pop.user_daily_rate
        self._user_cdf = np.cumsum(rates) / rates.sum()
        self._legit_rate = float(rates.sum()) / DAY  # events per second at peak-normalised load
        self._diurnal = np.array(DIURNAL_PROFILE) / max(DIURNAL_PROFILE)
        ato_rates = np.where(
            pop.user_risky,
            config.ato_rate_per_user_day * config.risky_user_ato_multiplier,
            config.ato_rate_per_user_day,
        )
        self._ato_cdf = np.cumsum(ato_rates) / ato_rates.sum()
        self._ato_rate = float(ato_rates.sum()) / DAY

    def events(self, start: float, end: float | None = None) -> Iterator[RawTransaction]:
        """Yield transactions with ``start <= timestamp < end`` in timestamp order."""
        rng = self._rng
        # The legitimate process is generated at its peak rate and thinned by the diurnal
        # profile; mean diurnal intensity is < 1, so scale the peak rate to keep the
        # configured average number of transactions per user and day.
        peak_rate = self._legit_rate / float(self._diurnal.mean())
        next_legit = start + rng.exponential(1.0 / peak_rate)
        next_ato = start + rng.exponential(1.0 / self._ato_rate)
        scheduled: list[tuple[float, int, RawTransaction]] = []
        sequence = 0

        while True:
            next_scheduled = scheduled[0][0] if scheduled else math.inf
            now = min(next_legit, next_ato, next_scheduled)
            if end is not None and now >= end:
                return
            if now == next_scheduled:
                yield heapq.heappop(scheduled)[2]
            elif now == next_ato:
                for tx in self._ato_burst(now):
                    heapq.heappush(scheduled, (tx.timestamp, sequence, tx))
                    sequence += 1
                next_ato += rng.exponential(1.0 / self._ato_rate)
            else:
                hour = int((now % DAY) // 3_600)
                if rng.random() < self._diurnal[hour]:
                    yield self._ordinary(now)
                next_legit += rng.exponential(1.0 / peak_rate)

    # -- transaction factories -------------------------------------------------------

    def _ordinary(self, ts: float) -> RawTransaction:
        rng = self._rng
        user = int(np.searchsorted(self._user_cdf, rng.random(), side="right"))
        user = min(user, len(self.population.user_ids) - 1)
        if rng.random() < self.config.opportunistic_fraud_prob:
            return self._opportunistic_fraud(user, ts)
        return self._legit(user, ts)

    def _legit(self, user: int, ts: float) -> RawTransaction:
        rng = self._rng
        pop = self.population
        home = pop.user_country[user]
        country = home
        if rng.random() < self.config.travel_prob:
            country = self._foreign_country(home)
            local = pop.merchants_by_country.get(country)
            merchant = int(rng.choice(local)) if local is not None else self._any_merchant()
        elif rng.random() < 0.75:
            merchant = int(rng.choice(pop.user_favourites[user]))
        else:
            merchant = int(rng.choice(pop.merchants_by_country[home]))

        amount = float(rng.lognormal(pop.user_log_mean[user], pop.user_log_sigma[user]))
        if rng.random() < self.config.large_purchase_prob:
            amount *= rng.uniform(4.0, 15.0)
        category = pop.merchant_category[merchant]
        card_present = (
            category not in ONLINE_CATEGORIES and rng.random() < pop.user_card_present_p[user]
        )
        device = pop.user_device[user] if rng.random() < 0.95 else self._new_device()
        return self._transaction(user, merchant, amount, country, card_present, device, ts, False)

    def _opportunistic_fraud(self, user: int, ts: float) -> RawTransaction:
        rng = self._rng
        pop = self.population
        merchant = (
            int(rng.choice(pop.high_risk_merchants)) if rng.random() < 0.7 else self._any_merchant()
        )
        home = pop.user_country[user]
        country = self._foreign_country(home) if rng.random() < 0.4 else home
        amount = self._typical_amount(user) * float(rng.lognormal(1.0, 0.7))
        return self._transaction(
            user, merchant, amount, country, False, self._new_device(), ts, True
        )

    def _ato_burst(self, start: float) -> list[RawTransaction]:
        rng = self._rng
        pop = self.population
        user = int(np.searchsorted(self._ato_cdf, rng.random(), side="right"))
        user = min(user, len(pop.user_ids) - 1)
        home = pop.user_country[user]
        country = self._foreign_country(home) if rng.random() < 0.7 else home
        device = self._new_device()
        n_tests = int(rng.integers(0, 3))
        n_spend = 1 + int(min(rng.poisson(3.0), 10))

        burst = []
        ts = start
        for i in range(n_tests + n_spend):
            if i < n_tests:
                amount = float(rng.uniform(0.5, 3.0))
                merchant = self._high_risk_or_any(0.8)
            else:
                amount = self._typical_amount(user) * float(rng.lognormal(1.6, 0.6))
                merchant = self._high_risk_or_any(0.6)
            card_present = rng.random() < 0.05
            burst.append(
                self._transaction(user, merchant, amount, country, card_present, device, ts, True)
            )
            ts += float(rng.exponential(240.0)) + 5.0
        return burst

    # -- helpers ---------------------------------------------------------------------

    def _transaction(
        self,
        user: int,
        merchant: int,
        amount: float,
        country: str,
        card_present: bool,
        device: str,
        ts: float,
        is_fraud: bool,
    ) -> RawTransaction:
        pop = self.population
        return RawTransaction(
            transaction_id=str(uuid.UUID(bytes=self._rng.bytes(16), version=4)),
            user_id=pop.user_ids[user],
            merchant_id=pop.merchant_ids[merchant],
            merchant_category=pop.merchant_category[merchant],
            amount=round(min(max(amount, 0.5), 25_000.0), 2),
            country=country,
            card_present=card_present,
            timestamp=round(ts, 3),
            device_id=device,
            is_fraud=is_fraud,
        )

    def _typical_amount(self, user: int) -> float:
        return float(np.exp(self.population.user_log_mean[user]))

    def _foreign_country(self, home: str) -> str:
        while True:
            country = str(self._rng.choice(ALL_COUNTRIES))
            if country != home:
                return country

    def _any_merchant(self) -> int:
        return int(self._rng.integers(0, len(self.population.merchant_ids)))

    def _high_risk_or_any(self, p_high_risk: float) -> int:
        if self._rng.random() < p_high_risk:
            return int(self._rng.choice(self.population.high_risk_merchants))
        return self._any_merchant()

    def _new_device(self) -> str:
        return f"d{self._rng.integers(0, 2**48):012x}"


def _stable_hash(text: str) -> int:
    """Process-independent hash (``hash()`` is salted per interpreter)."""
    return int.from_bytes(uuid.uuid5(uuid.NAMESPACE_OID, text).bytes[:8], "big")
