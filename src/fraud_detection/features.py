"""Online feature engineering over a Redis feature store.

Design notes
------------
* **Event time.** Every window is evaluated relative to the transaction's own
  ``timestamp``, never the wall clock, so replaying history offline (training data)
  and processing the live stream produce identical features for identical input.
* **Exact windows.** A user's recent transactions live in one sorted set scored by
  event time. Members encode ``transaction_id|amount|merchant_id|country|timestamp`` so
  a single ``ZRANGEBYSCORE`` yields counts, sums, distinct merchants/countries and amount
  statistics for every window.
* **Daily merchant risk.** Merchant fraud rates only use complete days whose labels are
  known, so they are final once computed; they are cached per merchant and day, the way
  production systems serve daily batch risk scores.
* **Idempotent writes.** All writes are ``ZADD``/``PFADD`` of values derived from the
  transaction itself, so re-processing a redelivered message leaves the store unchanged
  and recomputes the same features (at-least-once delivery is safe).
* **Delayed labels.** Fraud labels (chargebacks) are only usable once they would be
  known. ``label_delay_seconds`` shifts every label-derived window into the past, which
  prevents a fraud burst from "seeing" the labels of its own earlier transactions.
* **One round trip.** Writes never depend on computed features, so a whole batch of
  transactions is pipelined into a single Redis round trip while preserving sequential
  semantics (Redis executes pipelined commands in order).
* **Cluster friendly.** Per-entity keys use ``{hash tags}`` so multi-key commands stay
  within one slot on Redis Cluster.
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Final, Protocol

from fraud_detection.schemas import FeatureEvent, FeatureVector, RawTransaction

FEATURE_COLUMNS: Final[tuple[str, ...]] = tuple(FeatureVector.model_fields)

HOUR: Final = 3_600
DAY: Final = 86_400
USER_HISTORY_SECONDS: Final = 30 * DAY
MAX_ABS_ZSCORE: Final = 50.0


class RedisPipeline(Protocol):
    def zadd(self, name: str, mapping: dict[str, float]) -> Any: ...
    def zremrangebyscore(self, name: str, min: float | str, max: float | str) -> Any: ...
    def zrangebyscore(
        self, name: str, min: float | str, max: float | str, *, withscores: bool = ...
    ) -> Any: ...
    def zcount(self, name: str, min: float | str, max: float | str) -> Any: ...
    def pfadd(self, name: str, *values: str) -> Any: ...
    def pfcount(self, *sources: str) -> Any: ...
    def expire(self, name: str, time: int) -> Any: ...
    def execute(self) -> list[Any]: ...


class RedisClient(Protocol):
    def pipeline(self, transaction: bool = ..., shard_hint: str | None = ...) -> Any: ...


@dataclass(frozen=True)
class FeatureStoreConfig:
    key_prefix: str = "fd"
    label_delay_seconds: int = 2 * DAY
    merchant_window_days: int = 30
    merchant_prior_rate: float = 0.01
    merchant_prior_weight: float = 50.0

    @property
    def user_retention_seconds(self) -> int:
        """How far back user state is kept: history window plus the label delay."""
        return USER_HISTORY_SECONDS + self.label_delay_seconds

    @property
    def merchant_ttl_seconds(self) -> int:
        """Garbage-collection TTL for per-day merchant counters (wall-clock seconds)."""
        return (self.merchant_window_days + 2) * DAY + self.label_delay_seconds

    def user_key(self, user_id: str, kind: str) -> str:
        return f"{self.key_prefix}:u:{{{user_id}}}:{kind}"

    def merchant_key(self, merchant_id: str, kind: str, day: int) -> str:
        return f"{self.key_prefix}:m:{{{merchant_id}}}:{kind}:{day}"

    def clock_key(self) -> str:
        return f"{self.key_prefix}:sim:clock"


@dataclass(frozen=True)
class _Plan:
    history: int
    user_frauds: int
    merchant_key: tuple[str, int]
    merchant_tx: int | None
    merchant_fraud: int | None


def encode_member(tx: RawTransaction) -> str:
    """Sorted-set member holding everything the rolling windows need."""
    return f"{tx.transaction_id}|{tx.amount:.2f}|{tx.merchant_id}|{tx.country}|{tx.timestamp!r}"


def utc_day(timestamp: float) -> int:
    """Whole UTC days since the Unix epoch."""
    return int(timestamp // DAY)


def merchant_window_days(timestamp: float, config: FeatureStoreConfig) -> list[int]:
    """The complete UTC days whose labels are known at ``timestamp``."""
    last_known_day = utc_day(timestamp - config.label_delay_seconds) - 1
    return list(range(last_known_day - config.merchant_window_days + 1, last_known_day + 1))


def build_feature_vector(
    tx: RawTransaction,
    history: Sequence[bytes | str],
    known_user_frauds: int,
    merchant_tx_count: int,
    merchant_fraud_count: int,
    config: FeatureStoreConfig,
) -> FeatureVector:
    """Compute features from a store snapshot. Pure function: no I/O.

    ``history`` holds encoded members (see ``encode_member``) of the user's retained
    transactions, including ``tx`` itself.
    """
    ts = tx.timestamp
    delay = config.label_delay_seconds

    count_1h = count_24h = 0
    sum_1h = sum_24h = 0.0
    merchants_24h: set[str] = set()
    countries_7d: set[str] = set()
    prior_amounts: list[float] = []
    prior_countries: set[str] = set()
    last_prior_ts: float | None = None
    labelled_tx = 0

    for member in history:
        text = member.decode("utf-8") if isinstance(member, bytes) else member
        tx_id, amount_text, merchant_id, country, timestamp_text = text.split("|")
        entry_ts = float(timestamp_text)
        age = ts - entry_ts
        if age < 0:  # out-of-order: a later event already stored; not visible yet
            continue
        entry_amount = float(amount_text)
        if age < HOUR:
            count_1h += 1
            sum_1h += entry_amount
        if age < DAY:
            count_24h += 1
            sum_24h += entry_amount
            merchants_24h.add(merchant_id)
        if age < 7 * DAY:
            countries_7d.add(country)
        if tx_id != tx.transaction_id and age <= USER_HISTORY_SECONDS:
            prior_amounts.append(entry_amount)
            prior_countries.add(country)
            if last_prior_ts is None or entry_ts > last_prior_ts:
                last_prior_ts = entry_ts
        # Strictly older than the delay, so a transaction never sees its own label.
        if delay < age <= delay + USER_HISTORY_SECONDS:
            labelled_tx += 1

    amount = round(tx.amount, 2)
    if len(prior_amounts) >= 2:
        mean = math.fsum(prior_amounts) / len(prior_amounts)
        variance = math.fsum((a - mean) ** 2 for a in prior_amounts) / len(prior_amounts)
        zscore = (amount - mean) / max(math.sqrt(variance), 1.0)
        zscore = max(-MAX_ABS_ZSCORE, min(MAX_ABS_ZSCORE, zscore))
    else:
        zscore = 0.0

    seconds_since_last = (
        float(USER_HISTORY_SECONDS) if last_prior_ts is None else max(0.0, ts - last_prior_ts)
    )
    merchant_rate = (
        (merchant_fraud_count + config.merchant_prior_weight * config.merchant_prior_rate)
        / (merchant_tx_count + config.merchant_prior_weight)
        if merchant_tx_count + config.merchant_prior_weight > 0
        else 0.0
    )
    user_rate = min(1.0, known_user_frauds / labelled_tx) if labelled_tx else 0.0
    moment = datetime.fromtimestamp(ts, tz=UTC)

    return FeatureVector(
        amount=amount,
        amount_log=math.log1p(amount),
        amount_zscore=zscore,
        tx_count_1h=count_1h,
        tx_count_24h=count_24h,
        tx_sum_1h=round(sum_1h, 2),
        tx_sum_24h=round(sum_24h, 2),
        unique_merchants_24h=len(merchants_24h),
        unique_countries_7d=len(countries_7d),
        seconds_since_last_tx=seconds_since_last,
        is_new_country=bool(prior_countries) and tx.country not in prior_countries,
        hour_of_day=moment.hour,
        day_of_week=moment.weekday(),
        is_weekend=moment.weekday() >= 5,
        card_present=tx.card_present,
        merchant_fraud_rate_30d=min(1.0, merchant_rate),
        user_chargeback_rate=user_rate,
    )


class FeatureEngineer:
    """Maintains per-user/per-merchant state in Redis and emits feature events."""

    MERCHANT_CACHE_SIZE: Final = 200_000

    def __init__(self, redis_client: RedisClient, config: FeatureStoreConfig | None = None):
        self._redis = redis_client
        self.config = config or FeatureStoreConfig()
        # (merchant_id, last known day) -> (transactions, frauds) over the window.
        self._merchant_counts: dict[tuple[str, int], tuple[int, int]] = {}

    def compute(self, tx: RawTransaction) -> FeatureEvent:
        return self.compute_batch([tx])[0]

    def compute_batch(self, txs: Sequence[RawTransaction]) -> list[FeatureEvent]:
        """Update state and compute features for ``txs`` in order, in one round trip."""
        if not txs:
            return []
        if len(self._merchant_counts) > self.MERCHANT_CACHE_SIZE:
            self._merchant_counts.clear()
        pipe = self._redis.pipeline(transaction=False)
        counter = _Counter()
        plans = [self._queue(pipe, tx, counter) for tx in txs]
        results = pipe.execute()
        for plan in plans:
            if plan.merchant_tx is not None and plan.merchant_fraud is not None:
                self._merchant_counts[plan.merchant_key] = (
                    int(results[plan.merchant_tx]),
                    int(results[plan.merchant_fraud]),
                )
        processed_at = time.time()
        events = []
        for tx, plan in zip(txs, plans, strict=True):
            merchant_tx, merchant_fraud = self._merchant_counts[plan.merchant_key]
            vector = build_feature_vector(
                tx,
                results[plan.history],
                known_user_frauds=int(results[plan.user_frauds]),
                merchant_tx_count=merchant_tx,
                merchant_fraud_count=merchant_fraud,
                config=self.config,
            )
            events.append(
                FeatureEvent(
                    transaction_id=tx.transaction_id,
                    user_id=tx.user_id,
                    merchant_id=tx.merchant_id,
                    event_time=tx.timestamp,
                    emitted_at=tx.emitted_at,
                    processed_at=processed_at,
                    label=None if tx.is_fraud is None else int(tx.is_fraud),
                    features=vector,
                )
            )
        return events

    def _queue(self, pipe: RedisPipeline, tx: RawTransaction, counter: _Counter) -> _Plan:
        cfg = self.config
        ts = tx.timestamp
        retention = cfg.user_retention_seconds
        oldest = ts - retention
        user_ttl = retention + DAY

        tx_key = cfg.user_key(tx.user_id, "tx")
        counter.add(pipe.zadd(tx_key, {encode_member(tx): ts}))
        counter.add(pipe.zremrangebyscore(tx_key, "-inf", f"({oldest}"))
        counter.add(pipe.expire(tx_key, user_ttl))
        history = counter.add(pipe.zrangebyscore(tx_key, oldest, ts))

        fraud_key = cfg.user_key(tx.user_id, "fraud")
        if tx.is_fraud:
            counter.add(pipe.zadd(fraud_key, {tx.transaction_id: ts}))
            counter.add(pipe.zremrangebyscore(fraud_key, "-inf", f"({oldest}"))
            counter.add(pipe.expire(fraud_key, user_ttl))
        user_frauds = counter.add(
            pipe.zcount(fraud_key, oldest, f"({ts - cfg.label_delay_seconds}")
        )

        today = utc_day(ts)
        merchant_tx_key = cfg.merchant_key(tx.merchant_id, "tx", today)
        counter.add(pipe.pfadd(merchant_tx_key, tx.transaction_id))
        counter.add(pipe.expire(merchant_tx_key, cfg.merchant_ttl_seconds))
        if tx.is_fraud:
            merchant_fraud_key = cfg.merchant_key(tx.merchant_id, "fraud", today)
            counter.add(pipe.pfadd(merchant_fraud_key, tx.transaction_id))
            counter.add(pipe.expire(merchant_fraud_key, cfg.merchant_ttl_seconds))
        days = merchant_window_days(ts, cfg)
        merchant_key = (tx.merchant_id, days[-1])
        if merchant_key in self._merchant_counts:
            return _Plan(history, user_frauds, merchant_key, None, None)
        merchant_tx = counter.add(
            pipe.pfcount(*[cfg.merchant_key(tx.merchant_id, "tx", day) for day in days])
        )
        merchant_fraud = counter.add(
            pipe.pfcount(*[cfg.merchant_key(tx.merchant_id, "fraud", day) for day in days])
        )
        return _Plan(history, user_frauds, merchant_key, merchant_tx, merchant_fraud)


class _Counter:
    """Tracks the position of each queued pipeline command in the result list."""

    def __init__(self) -> None:
        self._next = 0

    def add(self, _queued: object) -> int:
        position = self._next
        self._next += 1
        return position
