from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import fakeredis
import mlflow
import pandas as pd
import pytest
from mlflow.tracking import MlflowClient

from fraud_detection.bootstrap import replay
from fraud_detection.config import MonitorSettings
from fraud_detection.features import FEATURE_COLUMNS, FeatureEngineer
from fraud_detection.modeling import EnsembleScorer, to_matrix
from fraud_detection.monitor import (
    CATEGORICAL_FEATURES,
    MONITORED_COLUMNS,
    SEASONAL_FEATURES,
    ChampionReference,
    DriftMonitor,
    Reference,
    ReportSummary,
    build_report,
    decision_row,
    publish_to_mlflow,
    summarize,
)
from fraud_detection.registry import ModelRef
from fraud_detection.schemas import Decision, DecisionEvent, FeatureVector
from fraud_detection.simulation import DAY, SimulationConfig, TransactionSimulator

T0 = 1_767_225_600.0


@pytest.fixture(scope="module")
def scored_history(tiny_scorer: EnsembleScorer) -> pd.DataFrame:
    """Realistic decisions: simulated history through the real feature pipeline."""
    simulator = TransactionSimulator(SimulationConfig(seed=3, n_users=300, n_merchants=80))
    frame = replay(FeatureEngineer(fakeredis.FakeRedis()), simulator.events(T0, T0 + 12 * DAY), 500)
    frame["prediction"] = tiny_scorer.predict_proba(to_matrix(frame, FEATURE_COLUMNS))
    frame["model_version"] = "1"
    frame["decision"] = Decision.APPROVE.value
    return frame


@pytest.fixture
def settings(tmp_path: Path) -> MonitorSettings:
    return MonitorSettings(
        reports_dir=tmp_path / "reports", min_rows=50, report_interval_s=0.001, reports_keep=2
    )


def _monitor(
    settings: MonitorSettings, reference: pd.DataFrame | None, published: list[Any] | None = None
) -> DriftMonitor:
    ref = Reference("1", reference) if reference is not None else None

    def publisher(summary: ReportSummary, snapshot: Any, include_html: bool) -> None:
        if published is not None:
            published.append((summary, include_html))

    return DriftMonitor(settings, lambda: ref, publisher)


def _fill(monitor: DriftMonitor, rows: pd.DataFrame) -> None:
    for record in rows.to_dict("records"):
        monitor.window.append({str(key): value for key, value in record.items()})


def test_decision_row_flattens_features_and_score(feature_payload: Any) -> None:
    event = DecisionEvent(
        transaction_id="tx-1",
        user_id="u1",
        merchant_id="m1",
        event_time=T0,
        scored_at=T0 + 1,
        fraud_probability=0.7,
        decision=Decision.REVIEW,
        model_version="4",
        label=1,
        features=FeatureVector(**feature_payload()),
    )

    row = decision_row(event)

    assert row["prediction"] == 0.7
    assert row["label"] == 1
    assert row["model_version"] == "4"
    assert set(FEATURE_COLUMNS) <= set(row)


def test_boolean_features_are_monitored_as_categorical() -> None:
    assert set(CATEGORICAL_FEATURES) == {"is_new_country", "card_present"}


def test_seasonal_features_are_excluded_from_drift_detection() -> None:
    assert not {"hour_of_day", "day_of_week", "is_weekend"} & set(MONITORED_COLUMNS)
    assert set(MONITORED_COLUMNS) == (set(FEATURE_COLUMNS) - set(SEASONAL_FEATURES)) | {
        "prediction"
    }


def test_a_short_window_does_not_drift_on_calendar_features(
    settings: MonitorSettings, scored_history: pd.DataFrame
) -> None:
    """A live window covers one weekday and a few hours; that alone must not alarm."""
    shuffled = scored_history.sample(frac=1.0, random_state=5)
    reference = shuffled.iloc[: len(shuffled) // 2]
    window = shuffled.iloc[len(shuffled) // 2 :].head(1_500)
    window = window.assign(hour_of_day=3, day_of_week=6, is_weekend=True)
    monitor = _monitor(settings, reference)
    _fill(monitor, window)

    summary = monitor.report()

    assert summary is not None
    assert summary.drift_share < settings.drift_share_threshold


def test_report_waits_for_enough_rows(
    settings: MonitorSettings, scored_history: pd.DataFrame
) -> None:
    monitor = _monitor(settings, scored_history)
    _fill(monitor, scored_history.head(10))
    assert monitor.due() is False
    _fill(monitor, scored_history.head(60))
    assert monitor.due() is True


def test_report_skipped_without_champion_reference(
    settings: MonitorSettings, scored_history: pd.DataFrame
) -> None:
    monitor = _monitor(settings, None)
    _fill(monitor, scored_history.head(100))

    assert monitor.report() is None
    assert list(settings.reports_dir.glob("*.html")) == []


def test_same_distribution_has_little_drift(
    settings: MonitorSettings, scored_history: pd.DataFrame
) -> None:
    published: list[Any] = []
    # Random halves of one pool: identical distributions. (Consecutive periods would
    # differ for real: early simulated days are a cold start while windows fill up.)
    shuffled = scored_history.sample(frac=1.0, random_state=0)
    reference = shuffled.iloc[: len(shuffled) // 2]
    current = shuffled.iloc[len(shuffled) // 2 :].head(1_500)
    monitor = _monitor(settings, reference, published)
    _fill(monitor, current)

    summary = monitor.report()

    assert summary is not None
    assert summary.drift_share < settings.drift_share_threshold
    assert summary.html_path.exists()
    assert published[0][1] is True  # first report for a model version keeps the HTML
    assert {"precision", "recall", "roc_auc"} <= set(summary.quality)

    monitor.report()
    assert published[1][1] is False  # later drift-free reports store metrics only


def test_shifted_traffic_is_flagged_as_drift(
    settings: MonitorSettings, scored_history: pd.DataFrame
) -> None:
    published: list[Any] = []
    current = scored_history.sample(800, random_state=1).copy()
    for column in ("amount", "tx_sum_1h", "tx_sum_24h", "tx_count_1h", "tx_count_24h"):
        current[column] = current[column] * 8
    current["card_present"] = False
    current["prediction"] = (current["prediction"] + 0.5).clip(upper=1.0)
    monitor = _monitor(settings, scored_history, published)
    _fill(monitor, current)

    summary = monitor.report()

    assert summary is not None
    assert summary.drift_detected is True
    assert summary.drifted_columns >= 5


def test_single_class_window_skips_classification_metrics(
    settings: MonitorSettings, scored_history: pd.DataFrame
) -> None:
    legit_only = scored_history[scored_history["label"] == 0].sample(300, random_state=2)
    monitor = _monitor(settings, scored_history)
    _fill(monitor, legit_only)

    summary = monitor.report()  # Evidently raises KeyError on one class; must not crash

    assert summary is not None
    assert summary.quality == {}


def test_old_reports_are_pruned(settings: MonitorSettings, scored_history: pd.DataFrame) -> None:
    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    for stamp in (1, 2, 3):
        (settings.reports_dir / f"drift_report_{stamp}.html").write_text("old")
    monitor = _monitor(settings, scored_history)
    _fill(monitor, scored_history.sample(200, random_state=3))

    monitor.report()

    assert len(list(settings.reports_dir.glob("drift_report_*.html"))) == settings.reports_keep


def test_summarize_reads_drift_and_quality(scored_history: pd.DataFrame) -> None:
    snapshot = build_report(
        scored_history.head(2_000), scored_history.tail(2_000), 0.3, probas_threshold=0.5
    )

    share, drifted, quality = summarize(snapshot)

    assert 0.0 <= share <= 1.0
    assert drifted >= 0
    assert 0.0 <= quality["roc_auc"] <= 1.0


class TestChampionReference:
    def test_caches_and_follows_alias(
        self, settings: MonitorSettings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        versions = iter(["1", "2"])
        downloads: list[str] = []
        now = [0.0]

        def resolve(*_: Any) -> ModelRef:
            version = next(versions)
            return ModelRef("fraud-detector", version, f"run-{version}", f"models:/x/{version}")

        def download(run_id: str) -> pd.DataFrame:
            downloads.append(run_id)
            return pd.DataFrame({"a": [1]})

        monkeypatch.setattr("fraud_detection.monitor.resolve_alias", resolve)
        monkeypatch.setattr("fraud_detection.monitor.download_reference", download)
        provider = ChampionReference(
            settings.model_copy(update={"champion_refresh_s": 60.0}),
            None,  # type: ignore[arg-type]
            clock=lambda: now[0],
        )

        first = provider()
        now[0] = 30.0
        cached = provider()  # within the refresh period: no registry call
        now[0] = 120.0
        refreshed = provider()

        assert first is not None
        assert cached is first
        assert refreshed is not None
        assert refreshed.version == "2"
        assert downloads == ["run-1", "run-2"]

    def test_no_champion_yet(
        self, settings: MonitorSettings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("fraud_detection.monitor.resolve_alias", lambda *_: None)
        assert ChampionReference(settings, None)() is None  # type: ignore[arg-type]


def test_publish_to_mlflow_logs_metrics_and_artifacts(
    settings: MonitorSettings,
    scored_history: pd.DataFrame,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri(f"file://{tmp_path / 'mlruns'}")
    monitor = DriftMonitor(
        settings, lambda: Reference("1", scored_history), publish_to_mlflow(settings)
    )
    _fill(monitor, scored_history.sample(300, random_state=4))

    monitor.report()

    client = MlflowClient()
    experiment = client.get_experiment_by_name(settings.experiment_name)
    assert experiment is not None
    run = client.search_runs([experiment.experiment_id])[0]
    assert "drift_share" in run.data.metrics
    artifacts = {a.path for a in client.list_artifacts(run.info.run_id, "evidently")}
    assert "evidently/snapshot.json" in artifacts
    assert any(path.endswith(".html") for path in artifacts)
    os.environ.pop("MLFLOW_TRACKING_URI", None)
