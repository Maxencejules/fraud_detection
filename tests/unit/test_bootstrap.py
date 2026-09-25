from __future__ import annotations

from pathlib import Path

import fakeredis
import pandas as pd
import pytest

from fraud_detection.bootstrap import (
    IDENTIFIER_COLUMNS,
    main,
    replay,
    reset_feature_store,
    run_bootstrap,
)
from fraud_detection.config import BootstrapSettings
from fraud_detection.features import FEATURE_COLUMNS, FeatureEngineer
from fraud_detection.simulation import DAY, SimulationConfig, TransactionSimulator

END = 1_767_225_600.0 + 20 * DAY


def _settings(tmp_path: Path) -> BootstrapSettings:
    return BootstrapSettings(
        sim_users=200,
        sim_merchants=60,
        history_days=6,
        output_path=tmp_path / "features.parquet",
        batch_size=250,
    )


def test_bootstrap_writes_training_set_and_warms_the_store(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    client = fakeredis.FakeRedis()

    summary = run_bootstrap(settings, client, end=END)

    frame = pd.read_parquet(settings.output_path)
    assert list(frame.columns) == [*IDENTIFIER_COLUMNS, *FEATURE_COLUMNS]
    assert summary.rows == len(frame) > 1_000
    assert frame["event_time"].between(END - 6 * DAY, END).all()
    assert frame["event_time"].is_monotonic_increasing
    assert set(frame["label"].unique()) == {0, 1}
    assert float(client.get(settings.feature_store_config().clock_key()) or 0) == END


def test_offline_rows_match_the_online_feature_engineer(tmp_path: Path) -> None:
    """Training rows are exactly what the streaming processor would have computed."""
    config = SimulationConfig(seed=11, n_users=100, n_merchants=40)
    events = list(TransactionSimulator(config).events(END - 3 * DAY, END))

    offline = replay(FeatureEngineer(fakeredis.FakeRedis()), events, batch_size=64)
    streaming = FeatureEngineer(fakeredis.FakeRedis())
    online = [streaming.compute(tx).features.model_dump() for tx in events]

    pd.testing.assert_frame_equal(
        offline[list(FEATURE_COLUMNS)].reset_index(drop=True),
        pd.DataFrame(online)[list(FEATURE_COLUMNS)],
        check_dtype=False,
    )


def test_reset_removes_only_feature_store_keys() -> None:
    client = fakeredis.FakeRedis()
    client.set("fd:u:{u1}:tx", "x")
    client.set("fd:sim:clock", "1")
    client.set("unrelated", "keep")

    assert reset_feature_store(client, "fd") == 2
    assert client.keys("*") == [b"unrelated"]


def test_cli_offline_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output = tmp_path / "out.parquet"
    monkeypatch.setenv("OUTPUT_PATH", str(output))
    monkeypatch.setenv("HISTORY_DAYS", "2")
    monkeypatch.setenv("SIM_USERS", "50")
    monkeypatch.setenv("SIM_MERCHANTS", "20")

    main(["--offline", "--end", str(END)])

    assert len(pd.read_parquet(output)) > 0


def test_interrupted_bootstrap_is_redone_from_a_clean_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "features.parquet"
    for name, value in {
        "OUTPUT_PATH": str(output),
        "HISTORY_DAYS": "2",
        "SIM_USERS": "50",
        "SIM_MERCHANTS": "20",
    }.items():
        monkeypatch.setenv(name, value)
    settings = BootstrapSettings()
    store = fakeredis.FakeRedis()
    # An earlier run with another end time died before writing its completion marker.
    run_bootstrap(settings, store, end=END - DAY / 2)
    store.delete(settings.feature_store_config().clock_key())
    output.unlink()
    monkeypatch.setattr("fraud_detection.bootstrap._connect", lambda *_: store)

    main(["--skip-if-present", "--end", str(END)])

    run_bootstrap(
        settings.model_copy(update={"output_path": tmp_path / "clean.parquet"}),
        fakeredis.FakeRedis(),
        end=END,
    )
    pd.testing.assert_frame_equal(
        pd.read_parquet(output), pd.read_parquet(tmp_path / "clean.parquet")
    )
