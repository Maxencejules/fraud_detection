import json
import os
import time

import pandas as pd

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

FEATURE_COLS = [
    "amount",
    "amount_log",
    "amount_zscore",
    "tx_count_1h",
    "tx_count_24h",
    "tx_sum_1h",
    "tx_sum_24h",
    "unique_merchants_24h",
    "unique_countries_7d",
    "hour_of_day",
    "day_of_week",
    "is_weekend",
    "card_present",
    "merchant_fraud_rate_30d",
    "user_chargeback_rate",
]


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
            print(f"Warning: Reference data not found at {REFERENCE_PATH}")
            self.reference_df = None

        os.makedirs(EVIDENTLY_REPORT_DIR, exist_ok=True)

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment("fraud-monitoring")

    def run_report(self):
        if self.reference_df is None or not self.buffer:
            return

        print(f"Running drift report for {len(self.buffer)} samples...")
        current_df = pd.DataFrame(self.buffer[:BATCH_SIZE])
        self.buffer = self.buffer[BATCH_SIZE:]

        predictions = []
        for tx_id in current_df["transaction_id"]:
            cache = self.r.get(f"pred:{tx_id}")
            predictions.append(float(cache.split(":")[0]) if cache else 0.0)

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
            numerical_features=[c for c in FEATURE_COLS if c not in ["is_weekend", "card_present"]],
            categorical_features=["is_weekend", "card_present", "day_of_week"],
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
                    mlflow.log_metric("dataset_drift_share", metric["result"]["drift_share"])
                    break

        self.last_report_time = time.time()
        print(f"Report saved and logged: {report_path}")


def main():
    if Consumer is None:
        raise RuntimeError("monitor service requires confluent-kafka. Install service dependencies before running it.")

    monitor = DriftMonitor()

    consumer_conf = {
        "bootstrap.servers": KAFKA_BOOTSTRAP_SERVERS,
        "group.id": "evidently-monitor",
        "auto.offset.reset": "latest",
    }
    consumer = Consumer(consumer_conf)
    consumer.subscribe([TOPIC_FEATURES])

    print(f"Monitoring service started. Consuming from {TOPIC_FEATURES}...")

    try:
        while True:
            msg = consumer.poll(1.0)

            if msg is not None and not msg.error():
                try:
                    monitor.buffer.append(json.loads(msg.value().decode("utf-8")))
                except Exception as exc:
                    print(f"Error parsing message: {exc}")

            elapsed = time.time() - monitor.last_report_time
            if (len(monitor.buffer) >= BUFFER_THRESHOLD and elapsed >= REPORT_INTERVAL_S) or (
                len(monitor.buffer) >= BATCH_SIZE
            ):
                monitor.run_report()

    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        consumer.close()


if __name__ == "__main__":
    main()
