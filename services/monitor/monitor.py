import json
import logging
import os
import time

import pandas as pd
from prometheus_client import Counter, Gauge, Histogram, start_http_server

try:
    import mlflow
except ImportError:  # pragma: no cover - runtime dependency only
    mlflow = None

try:
    import redis
except ImportError:  # pragma: no cover - runtime dependency only
    redis = None

try:
    from confluent_kafka import Consumer
except ImportError:  # pragma: no cover - runtime dependency only
    Consumer = None

try:
    from evidently import ColumnMapping
    from evidently.metric_preset import ClassificationPreset, DataDriftPreset, DataQualityPreset
    from evidently.report import Report
except ImportError:  # pragma: no cover - runtime dependency only
    ColumnMapping = ClassificationPreset = DataDriftPreset = DataQualityPreset = Report = None

from shared.config import CATEGORICAL_FEATURES, MONITOR_EXPERIMENT_NAME, NUMERICAL_FEATURES
from shared.observability import configure_logging, log_event

# Environment Variables
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
TOPIC_FEATURES = os.getenv("TOPIC_FEATURES", "transactions.features")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
REPORT_INTERVAL_S = int(os.getenv("REPORT_INTERVAL_S", "300"))
BUFFER_THRESHOLD = int(os.getenv("BUFFER_THRESHOLD", "100"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "2000"))
REFERENCE_PATH = os.getenv("REFERENCE_PATH", "/app/data/reference.parquet")
EVIDENTLY_REPORT_DIR = os.getenv("EVIDENTLY_REPORT_DIR", "/app/reports")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
METRICS_PORT = int(os.getenv("METRICS_PORT", "9109"))

logger = configure_logging("monitor")

REPORTS_CREATED = Counter(
    "fraud_monitor_reports_created_total",
    "Number of Evidently drift reports created.",
)
MONITOR_ERRORS = Counter(
    "fraud_monitor_errors_total",
    "Number of monitoring errors grouped by stage.",
    ["stage"],
)
REDIS_CACHE_MISSES = Counter(
    "fraud_monitor_prediction_cache_misses_total",
    "Number of missing prediction cache entries observed while building reports.",
)
BUFFER_SIZE = Gauge(
    "fraud_monitor_buffer_size",
    "Current number of feature events buffered for drift analysis.",
)
LAST_DRIFT_SHARE = Gauge(
    "fraud_monitor_last_dataset_drift_share",
    "Last observed dataset drift share from Evidently.",
)
REPORT_DURATION = Histogram(
    "fraud_monitor_report_duration_seconds",
    "Time spent generating and logging a drift report.",
    buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
)


def _start_metrics_server() -> None:
    if METRICS_PORT <= 0:
        return

    start_http_server(METRICS_PORT)
    log_event(logger, logging.INFO, "metrics_server_started", metrics_port=METRICS_PORT)


class DriftMonitor:
    def __init__(self):
        if redis is None or mlflow is None or Report is None:
            raise RuntimeError(
                "monitor service requires redis, mlflow, evidently, and confluent-kafka dependencies."
            )

        self.r = redis.from_url(REDIS_URL, decode_responses=True)
        self.buffer = []
        self.last_report_time = time.time()

        if os.path.exists(REFERENCE_PATH):
            self.reference_df = pd.read_parquet(REFERENCE_PATH)
            if "prediction" not in self.reference_df.columns:
                self.reference_df["prediction"] = self.reference_df["label"].astype(float)
        else:
            log_event(logger, logging.WARNING, "reference_data_missing", reference_path=REFERENCE_PATH)
            self.reference_df = None

        os.makedirs(EVIDENTLY_REPORT_DIR, exist_ok=True)

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MONITOR_EXPERIMENT_NAME)

    def run_report(self):
        if self.reference_df is None or not self.buffer:
            return

        report_start = time.perf_counter()
        batch_size = min(len(self.buffer), BATCH_SIZE)
        log_event(logger, logging.INFO, "drift_report_started", batch_size=batch_size)
        current_df = pd.DataFrame(self.buffer[:BATCH_SIZE])
        self.buffer = self.buffer[BATCH_SIZE:]
        BUFFER_SIZE.set(len(self.buffer))

        predictions = []
        for tx_id in current_df["transaction_id"]:
            cache = self.r.get(f"pred:{tx_id}")
            if cache:
                predictions.append(float(cache.split(":")[0]))
            else:
                REDIS_CACHE_MISSES.inc()
                predictions.append(0.0)

        current_df["prediction"] = predictions

        report = Report(
            metrics=[
                DataDriftPreset(drift_share=0.3),
                ClassificationPreset(),
                DataQualityPreset(),
            ]
        )

        column_mapping = ColumnMapping(
            target="label",
            prediction="prediction",
            numerical_features=NUMERICAL_FEATURES,
            categorical_features=CATEGORICAL_FEATURES,
        )

        report.run(
            reference_data=self.reference_df,
            current_data=current_df,
            column_mapping=column_mapping,
        )

        ts = int(time.time())
        report_path = os.path.join(EVIDENTLY_REPORT_DIR, f"drift_report_{ts}.html")
        report.save_html(report_path)

        with mlflow.start_run(run_name=f"drift_report_{ts}"):
            mlflow.log_artifact(report_path)

            result = report.as_dict()
            for metric in result.get("metrics", []):
                if metric.get("metric") == "DatasetDriftMetric":
                    drift_share = metric["result"]["drift_share"]
                    LAST_DRIFT_SHARE.set(drift_share)
                    mlflow.log_metric("dataset_drift_share", drift_share)
                    break

        self.last_report_time = time.time()
        REPORTS_CREATED.inc()
        REPORT_DURATION.observe(time.perf_counter() - report_start)
        log_event(logger, logging.INFO, "drift_report_completed", report_path=report_path)


def main():
    if Consumer is None:
        raise RuntimeError("monitor service requires confluent-kafka. Install service dependencies before running it.")

    _start_metrics_server()
    monitor = DriftMonitor()

    consumer_conf = {
        "bootstrap.servers": KAFKA_BOOTSTRAP_SERVERS,
        "group.id": "evidently-monitor",
        "auto.offset.reset": "latest",
    }
    consumer = Consumer(consumer_conf)
    consumer.subscribe([TOPIC_FEATURES])

    log_event(logger, logging.INFO, "monitor_started", features_topic=TOPIC_FEATURES)

    try:
        while True:
            msg = consumer.poll(1.0)

            if msg is not None and not msg.error():
                try:
                    monitor.buffer.append(json.loads(msg.value().decode("utf-8")))
                    BUFFER_SIZE.set(len(monitor.buffer))
                except Exception as exc:
                    MONITOR_ERRORS.labels(stage="deserialize").inc()
                    log_event(logger, logging.ERROR, "feature_message_parse_failed", error=str(exc))
            elif msg is not None and msg.error():
                MONITOR_ERRORS.labels(stage="consume").inc()
                log_event(logger, logging.ERROR, "monitor_consumer_error", error=str(msg.error()))

            elapsed = time.time() - monitor.last_report_time
            if (len(monitor.buffer) >= BUFFER_THRESHOLD and elapsed >= REPORT_INTERVAL_S) or (
                len(monitor.buffer) >= BATCH_SIZE
            ):
                try:
                    monitor.run_report()
                except Exception as exc:
                    MONITOR_ERRORS.labels(stage="report").inc()
                    log_event(logger, logging.ERROR, "drift_report_failed", error=str(exc))

    except KeyboardInterrupt:
        log_event(logger, logging.INFO, "monitor_shutdown_requested")
    finally:
        consumer.close()


if __name__ == "__main__":
    main()
