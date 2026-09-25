from __future__ import annotations

from typing import Any

import pytest

from fraud_detection.kafka_utils import GracefulShutdown
from fraud_detection.producer import Pacer, publish
from fraud_detection.schemas import RawTransaction


class FakeClock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


def test_pacer_spaces_events_on_an_absolute_schedule() -> None:
    clock = FakeClock()
    pacer = Pacer(rate_per_s=4, clock=clock)

    assert pacer.delay() == 0.0
    assert pacer.delay() == pytest.approx(0.25)
    clock.now += 0.1  # work took 100 ms; the next slot is still 0.5 s after start
    assert pacer.delay() == pytest.approx(0.4)


def test_pacer_skips_ahead_instead_of_bursting_after_a_stall() -> None:
    clock = FakeClock()
    pacer = Pacer(rate_per_s=10, clock=clock)
    pacer.delay()
    clock.now += 5.0

    assert pacer.delay() == 0.0
    assert pacer.delay() == pytest.approx(0.1)


def _publish(kafka: Any, shutdown: GracefulShutdown, txs: list[RawTransaction]) -> Any:
    producer = kafka.Producer()
    saved: list[float] = []
    count = publish(
        producer,
        "transactions.raw",
        txs,
        rate_per_s=1_000_000,
        shutdown=shutdown,
        save_clock=saved.append,
    )
    return producer, saved, count


def test_publishes_with_emission_time_keyed_by_user(
    kafka: Any, shutdown: GracefulShutdown, make_tx: Any
) -> None:
    txs = [make_tx(user_id=f"u{i}", timestamp=1_767_225_600.0 + i) for i in range(3)]

    producer, saved, count = _publish(kafka, shutdown, txs)

    assert count == 3
    published = [RawTransaction.model_validate_json(v) for v in producer.values("transactions.raw")]
    assert [p.transaction_id for p in published] == [t.transaction_id for t in txs]
    assert all(p.emitted_at is not None for p in published)
    assert [k for _, k, _ in producer.messages] == ["u0", "u1", "u2"]
    assert saved[-1] == txs[-1].timestamp  # clock persisted for restarts


def test_retries_when_the_local_queue_is_full(
    kafka: Any, shutdown: GracefulShutdown, make_tx: Any
) -> None:
    producer = kafka.Producer()
    producer.buffer_errors = 2

    count = publish(
        producer, "t", [make_tx()], rate_per_s=1e6, shutdown=shutdown, save_clock=lambda _: None
    )

    assert count == 1
    assert len(producer.messages) == 1


def test_stops_when_shutdown_is_requested(
    kafka: Any, shutdown: GracefulShutdown, make_tx: Any
) -> None:
    shutdown.request()

    _producer, saved, count = _publish(kafka, shutdown, [make_tx(), make_tx()])

    assert count == 0
    assert saved == []
