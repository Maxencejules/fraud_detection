"""Simulate transaction history, backfill the online feature store, write the training set.

Training rows are produced by replaying simulated history through the very same
``FeatureEngineer`` the streaming feature processor uses, so offline (training) and
online (serving) features are computed by one implementation. When run against the
Redis instance used by the live pipeline, the store is left warm: the producer resumes
the simulation clock where the history ends and every user keeps their past behaviour.
"""

from __future__ import annotations

import argparse
import itertools
import logging
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
import redis

from fraud_detection.config import BootstrapSettings
from fraud_detection.features import FEATURE_COLUMNS, FeatureEngineer
from fraud_detection.observability import configure_logging
from fraud_detection.simulation import DAY, TransactionSimulator

if TYPE_CHECKING:
    from fraud_detection.schemas import RawTransaction

IDENTIFIER_COLUMNS = ("transaction_id", "user_id", "merchant_id", "event_time", "label")

logger = logging.getLogger("fraud_detection.bootstrap")


@dataclass(frozen=True)
class BootstrapSummary:
    rows: int
    fraud_rate: float
    start: float
    end: float
    output_path: Path
    elapsed_s: float


def replay(
    engineer: FeatureEngineer, transactions: Iterable[RawTransaction], batch_size: int
) -> pd.DataFrame:
    """Run ``transactions`` through the feature pipeline and return one row per event."""
    rows: list[dict[str, Any]] = []
    iterator = iter(transactions)
    while batch := list(itertools.islice(iterator, batch_size)):
        for event in engineer.compute_batch(batch):
            row: dict[str, Any] = {
                "transaction_id": event.transaction_id,
                "user_id": event.user_id,
                "merchant_id": event.merchant_id,
                "event_time": event.event_time,
                "label": event.label,
            }
            row.update(event.features.model_dump())
            rows.append(row)
    return pd.DataFrame(rows, columns=[*IDENTIFIER_COLUMNS, *FEATURE_COLUMNS])


def write_parquet_atomically(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    frame.to_parquet(tmp, index=False)
    tmp.replace(path)


def reset_feature_store(client: redis.Redis, prefix: str) -> int:
    """Delete every key owned by the feature store. Returns the number of keys removed."""
    removed = 0
    batch: list[Any] = []
    for key in client.scan_iter(match=f"{prefix}:*", count=1_000):
        batch.append(key)
        if len(batch) >= 1_000:
            removed += int(client.unlink(*batch))
            batch.clear()
    if batch:
        removed += int(client.unlink(*batch))
    return removed


def run_bootstrap(
    settings: BootstrapSettings,
    client: redis.Redis,
    *,
    end: float,
    reset: bool = False,
) -> BootstrapSummary:
    started = time.perf_counter()
    store_config = settings.feature_store_config()
    if reset:
        removed = reset_feature_store(client, store_config.key_prefix)
        logger.info("feature_store_reset", extra={"keys_removed": removed})

    start = end - settings.history_days * DAY
    simulator = TransactionSimulator(settings.simulation_config(), stream="history")
    engineer = FeatureEngineer(client, store_config)
    frame = replay(engineer, simulator.events(start, end), settings.batch_size)
    write_parquet_atomically(frame, settings.output_path)
    client.set(store_config.clock_key(), repr(end))

    return BootstrapSummary(
        rows=len(frame),
        fraud_rate=float(frame["label"].mean()) if len(frame) else 0.0,
        start=start,
        end=end,
        output_path=settings.output_path,
        elapsed_s=time.perf_counter() - started,
    )


def _connect(settings: BootstrapSettings, offline: bool) -> redis.Redis:
    if offline:
        import fakeredis

        fake: redis.Redis = fakeredis.FakeRedis()
        return fake

    client = redis.Redis.from_url(settings.redis_url)
    deadline = time.monotonic() + 60
    while True:
        try:
            client.ping()
            return client
        except redis.ConnectionError:
            if time.monotonic() > deadline:
                raise
            time.sleep(1)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulate history, backfill the feature store and write the training set."
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="use in-memory Redis emulation instead of REDIS_URL (slower, nothing backfilled)",
    )
    parser.add_argument(
        "--reset", action="store_true", help="delete existing feature-store keys first"
    )
    parser.add_argument(
        "--skip-if-present",
        action="store_true",
        help="exit successfully if the store is already bootstrapped and the dataset exists",
    )
    parser.add_argument(
        "--end",
        type=float,
        default=None,
        help="end of the simulated history as a Unix timestamp (default: HISTORY_END or now)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    settings = BootstrapSettings()
    configure_logging("bootstrap", settings.log_level)
    client = _connect(settings, args.offline)

    if args.skip_if_present and not args.offline:
        clock_key = settings.feature_store_config().clock_key()
        if client.exists(clock_key) and settings.output_path.exists():
            logger.info("bootstrap_skipped", extra={"reason": "already bootstrapped"})
            return

    end = next(
        (value for value in (args.end, settings.history_end) if value is not None),
        float(int(time.time())),
    )
    logger.info(
        "bootstrap_started",
        extra={"history_days": settings.history_days, "end": end, "offline": args.offline},
    )
    summary = run_bootstrap(settings, client, end=end, reset=args.reset or args.offline)
    logger.info(
        "bootstrap_completed",
        extra={
            "rows": summary.rows,
            "fraud_rate": round(summary.fraud_rate, 5),
            "output_path": str(summary.output_path),
            "elapsed_s": round(summary.elapsed_s, 1),
        },
    )


if __name__ == "__main__":
    main()
