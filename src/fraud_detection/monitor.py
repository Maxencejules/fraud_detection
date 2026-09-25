"""Monitors data drift and live model quality with Evidently.

The monitor consumes the decisions topic, which carries each transaction's features,
the served probability and (in the simulation) the ground-truth label. Every
``report_interval_s`` it compares the most recent window with the *champion's own
reference data*: the held-out test period logged by the trainer together with the
model's predictions on it. When the champion changes, the reference changes with it.

Artifact retention: metrics and the compact JSON snapshot of every report go to MLflow;
the full HTML report (~4 MB) is uploaded only when drift is detected or for the first
report of a model version, and local HTML copies are pruned to ``reports_keep``.
"""

from __future__ import annotations

import logging
import tempfile
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
from confluent_kafka import Consumer
from evidently import BinaryClassification, DataDefinition, Dataset, Report
from evidently.presets import ClassificationPreset, DataDriftPreset
from mlflow.tracking import MlflowClient
from prometheus_client import Counter, Gauge, Histogram
from pydantic import ValidationError

from fraud_detection.config import MonitorSettings
from fraud_detection.features import FEATURE_COLUMNS
from fraud_detection.kafka_utils import GracefulShutdown, check_message_error, consumer_config
from fraud_detection.observability import configure_logging, start_metrics_server
from fraud_detection.registry import download_reference, resolve_alias, wait_for_tracking_server
from fraud_detection.schemas import DecisionEvent, FeatureVector

logger = logging.getLogger("fraud_detection.monitor")

# Calendar features are left out of drift detection. A monitoring window spans minutes of
# traffic while the reference spans days, so their distributions can never match (a
# window holds one weekday and a few hours): they would raise permanent false alarms.
SEASONAL_FEATURES = ("hour_of_day", "day_of_week", "is_weekend")
CATEGORICAL_FEATURES = tuple(
    name
    for name, field in FeatureVector.model_fields.items()
    if field.annotation is bool and name not in SEASONAL_FEATURES
)
NUMERICAL_FEATURES = tuple(
    c for c in FEATURE_COLUMNS if c not in CATEGORICAL_FEATURES and c not in SEASONAL_FEATURES
)
MONITORED_COLUMNS = (*NUMERICAL_FEATURES, *CATEGORICAL_FEATURES, "prediction")

REPORTS = Counter("fraud_monitor_reports_total", "Evidently reports generated.")
ERRORS = Counter("fraud_monitor_errors_total", "Monitoring failures, by stage.", ["stage"])
WINDOW_ROWS = Gauge("fraud_monitor_window_rows", "Decisions currently buffered.")
DRIFT_SHARE = Gauge("fraud_monitor_drift_share", "Share of monitored columns that drifted.")
DRIFT_DETECTED = Gauge("fraud_monitor_drift_detected", "1 if dataset drift was detected.")
QUALITY = Gauge("fraud_monitor_quality", "Live model quality on labelled decisions.", ["metric"])
REPORT_SECONDS = Histogram(
    "fraud_monitor_report_seconds",
    "Time to build and publish one report.",
    buckets=(0.5, 1, 2.5, 5, 10, 30, 60, 120),
)

QUALITY_METRICS = {
    "Precision": "precision",
    "Recall": "recall",
    "F1Score": "f1",
    "RocAuc": "roc_auc",
    "Accuracy": "accuracy",
}


@dataclass(frozen=True)
class Reference:
    version: str
    frame: pd.DataFrame


@dataclass(frozen=True)
class ReportSummary:
    rows: int
    drift_share: float
    drifted_columns: int
    drift_detected: bool
    quality: dict[str, float]
    model_versions: tuple[str, ...]
    html_path: Path


def decision_row(event: DecisionEvent) -> dict[str, Any]:
    row: dict[str, Any] = event.features.model_dump()
    row.update(
        transaction_id=event.transaction_id,
        event_time=event.event_time,
        label=event.label,
        prediction=event.fraud_probability,
        decision=event.decision.value,
        model_version=event.model_version,
    )
    return row


def build_report(
    reference: pd.DataFrame, current: pd.DataFrame, drift_share: float, probas_threshold: float
) -> Any:
    """Drift over features and scores; classification quality when both classes exist."""
    with_labels = (
        current["label"].notna().all()
        and current["label"].nunique() == 2
        and reference["label"].nunique() == 2
    )
    definition = DataDefinition(
        numerical_columns=[*NUMERICAL_FEATURES, "prediction"],
        categorical_columns=[*CATEGORICAL_FEATURES, "label"]
        if with_labels
        else [*CATEGORICAL_FEATURES],
        classification=(
            [BinaryClassification(target="label", prediction_probas="prediction", pos_label=1)]
            if with_labels
            else None
        ),
    )
    metrics: list[Any] = [DataDriftPreset(columns=list(MONITORED_COLUMNS), drift_share=drift_share)]
    if with_labels:
        metrics.append(ClassificationPreset(probas_threshold=probas_threshold))

    columns = [*MONITORED_COLUMNS, "label"] if with_labels else list(MONITORED_COLUMNS)
    current_frame = current[columns].astype({"label": int} if with_labels else {})
    reference_frame = reference[columns].astype({"label": int} if with_labels else {})
    return Report(metrics).run(
        Dataset.from_pandas(current_frame, data_definition=definition),
        Dataset.from_pandas(reference_frame, data_definition=definition),
    )


def summarize(snapshot: Any) -> tuple[float, int, dict[str, float]]:
    """Extract (drift share, drifted column count, quality metrics) from a snapshot."""
    share, drifted = 0.0, 0
    quality: dict[str, float] = {}
    for metric in snapshot.dict()["metrics"]:
        name: str = metric["metric_name"]
        kind = name.split("(", 1)[0]
        value = metric["value"]
        if kind == "DriftedColumnsCount":
            share, drifted = float(value["share"]), int(value["count"])
        elif kind in QUALITY_METRICS and isinstance(value, int | float):
            quality[QUALITY_METRICS[kind]] = float(value)
    return share, drifted, quality


class DriftMonitor:
    def __init__(
        self,
        settings: MonitorSettings,
        reference_provider: Callable[[], Reference | None],
        publisher: Callable[[ReportSummary, Any, bool], None] | None = None,
    ) -> None:
        self.settings = settings
        self.window: deque[dict[str, Any]] = deque(maxlen=settings.window_size)
        self._reference_provider = reference_provider
        self._publisher = publisher
        self._last_report = time.monotonic()
        self._reported_versions: set[str] = set()
        settings.reports_dir.mkdir(parents=True, exist_ok=True)

    def add(self, event: DecisionEvent) -> None:
        self.window.append(decision_row(event))
        WINDOW_ROWS.set(len(self.window))

    def due(self) -> bool:
        elapsed = time.monotonic() - self._last_report
        return (
            elapsed >= self.settings.report_interval_s
            and len(self.window) >= self.settings.min_rows
        )

    def report(self) -> ReportSummary | None:
        started = time.perf_counter()
        self._last_report = time.monotonic()
        reference = self._reference_provider()
        if reference is None:
            logger.info("report_skipped", extra={"reason": "no champion reference yet"})
            return None

        current = pd.DataFrame(list(self.window))
        snapshot = build_report(
            reference.frame,
            current,
            self.settings.drift_share_threshold,
            probas_threshold=self.settings.threshold_review,
        )
        share, drifted, quality = summarize(snapshot)
        html_path = self.settings.reports_dir / f"drift_report_{int(time.time())}.html"
        snapshot.save_html(str(html_path))
        self._prune_reports()

        summary = ReportSummary(
            rows=len(current),
            drift_share=share,
            drifted_columns=drifted,
            drift_detected=share >= self.settings.drift_share_threshold,
            quality=quality,
            model_versions=tuple(sorted(current["model_version"].unique())),
            html_path=html_path,
        )
        first_for_version = reference.version not in self._reported_versions
        self._reported_versions.add(reference.version)
        if self._publisher is not None:
            self._publisher(summary, snapshot, summary.drift_detected or first_for_version)

        REPORTS.inc()
        DRIFT_SHARE.set(share)
        DRIFT_DETECTED.set(int(summary.drift_detected))
        for name, value in quality.items():
            QUALITY.labels(metric=name).set(value)
        REPORT_SECONDS.observe(time.perf_counter() - started)
        logger.info(
            "report_completed",
            extra={
                "rows": summary.rows,
                "drift_share": round(share, 4),
                "drifted_columns": drifted,
                "drift_detected": summary.drift_detected,
                "reference_version": reference.version,
                **{k: round(v, 4) for k, v in quality.items()},
            },
        )
        return summary

    def _prune_reports(self) -> None:
        reports = sorted(self.settings.reports_dir.glob("drift_report_*.html"))
        for stale in reports[: -self.settings.reports_keep]:
            stale.unlink(missing_ok=True)


class ChampionReference:
    """Caches the champion's reference dataset; refreshes when the alias moves."""

    def __init__(
        self,
        settings: MonitorSettings,
        client: MlflowClient,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._settings = settings
        self._client = client
        self._clock = clock
        self._cached: Reference | None = None
        self._checked_at = 0.0

    def __call__(self) -> Reference | None:
        now = self._clock()
        if self._cached and now - self._checked_at < self._settings.champion_refresh_s:
            return self._cached
        self._checked_at = now
        ref = resolve_alias(self._client, self._settings.model_name, self._settings.model_alias)
        if ref is None or ref.run_id is None:
            return self._cached
        if self._cached is None or self._cached.version != ref.version:
            self._cached = Reference(ref.version, download_reference(ref.run_id))
            logger.info(
                "reference_loaded", extra={"version": ref.version, "rows": len(self._cached.frame)}
            )
        return self._cached


def publish_to_mlflow(settings: MonitorSettings) -> Callable[[ReportSummary, Any, bool], None]:
    def publish(summary: ReportSummary, snapshot: Any, include_html: bool) -> None:
        mlflow.set_experiment(settings.experiment_name)
        with (
            mlflow.start_run(run_name=summary.html_path.stem),
            tempfile.TemporaryDirectory() as tmp,
        ):
            mlflow.set_tags(
                {
                    "model_versions": ",".join(summary.model_versions),
                    "drift_detected": str(summary.drift_detected).lower(),
                }
            )
            mlflow.log_metrics(
                {
                    "window_rows": summary.rows,
                    "drift_share": summary.drift_share,
                    "drifted_columns": summary.drifted_columns,
                    **{f"live_{k}": v for k, v in summary.quality.items()},
                }
            )
            snapshot_path = Path(tmp) / "snapshot.json"
            snapshot.save_json(str(snapshot_path))
            mlflow.log_artifact(str(snapshot_path), artifact_path="evidently")
            if include_html:
                mlflow.log_artifact(str(summary.html_path), artifact_path="evidently")

    return publish


def main() -> None:
    settings = MonitorSettings()
    configure_logging("monitor", settings.log_level)
    start_metrics_server(settings.metrics_port, logger)
    shutdown = GracefulShutdown()
    wait_for_tracking_server(settings.mlflow_tracking_uri)
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

    monitor = DriftMonitor(
        settings,
        ChampionReference(settings, MlflowClient()),
        publish_to_mlflow(settings),
    )
    consumer = Consumer(
        consumer_config(
            settings.kafka_bootstrap_servers,
            settings.consumer_group,
            **{"enable.auto.offset.store": True, "auto.offset.reset": "latest"},
        )
    )
    consumer.subscribe([settings.topic_decisions])
    logger.info("monitor_started", extra={"topic": settings.topic_decisions})
    try:
        while not shutdown.requested:
            for msg in consumer.consume(num_messages=500, timeout=1.0):
                if not check_message_error(msg):
                    continue
                try:
                    monitor.add(DecisionEvent.model_validate_json(msg.value() or b""))
                except ValidationError:
                    ERRORS.labels(stage="deserialize").inc()
                    logger.warning("decision_parse_failed", extra={"offset": msg.offset()})
            if monitor.due():
                try:
                    monitor.report()
                except Exception:
                    ERRORS.labels(stage="report").inc()
                    logger.exception("report_failed")
    finally:
        consumer.close()
        logger.info("monitor_stopped")


if __name__ == "__main__":
    main()
