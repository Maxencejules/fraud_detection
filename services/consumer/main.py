import json
import logging
import math
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict

from prometheus_client import Counter, Histogram, start_http_server

try:
    import redis
except ImportError:  # pragma: no cover - exercised in lightweight test envs
    redis = None

try:
    from confluent_kafka import Consumer, KafkaError, Producer
    from confluent_kafka.admin import AdminClient, NewTopic
except ImportError:  # pragma: no cover - exercised in lightweight test envs
    Consumer = KafkaError = Producer = AdminClient = NewTopic = None

from shared.observability import configure_logging, log_event

# Environment Variables
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_GROUP_ID = os.getenv("KAFKA_GROUP_ID", "feature-engineer-group")
TOPIC_TRANSACTIONS = os.getenv("TOPIC_TRANSACTIONS", "transactions.raw")
TOPIC_FEATURES = os.getenv("TOPIC_FEATURES", "transactions.features")
TOPIC_DLQ = os.getenv("TOPIC_DLQ", f"{TOPIC_TRANSACTIONS}.dlq")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
METRICS_PORT = int(os.getenv("METRICS_PORT", "9108"))

logger = configure_logging("consumer")

MESSAGES_PROCESSED = Counter(
    "fraud_consumer_messages_processed_total",
    "Number of raw transactions successfully converted to feature events.",
)
MESSAGES_FAILED = Counter(
    "fraud_consumer_messages_failed_total",
    "Number of raw transactions that failed processing.",
    ["stage"],
)
MESSAGES_DLQ = Counter(
    "fraud_consumer_messages_dlq_total",
    "Number of raw transactions published to the dead-letter topic.",
)
FEATURE_COMPUTE_LATENCY = Histogram(
    "fraud_consumer_feature_compute_seconds",
    "Time spent computing features for a raw transaction.",
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 1.0, 5.0),
)


def _as_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _start_metrics_server() -> None:
    if METRICS_PORT <= 0:
        return

    start_http_server(METRICS_PORT)
    log_event(logger, logging.INFO, "metrics_server_started", metrics_port=METRICS_PORT)


def build_dead_letter_message(
    raw_value: Any,
    error: Exception,
    stage: str,
    topic: str,
    partition: int,
    offset: int,
    message_key: Any = None,
) -> str:
    payload = {
        "error": str(error),
        "message_key": _as_text(message_key) if message_key is not None else None,
        "offset": offset,
        "original_payload": _as_text(raw_value),
        "partition": partition,
        "source_topic": topic,
        "stage": stage,
        "ts_utc": datetime.now(timezone.utc).isoformat(),
    }
    return json.dumps(payload)


def publish_dead_letter(
    producer: Any,
    raw_value: Any,
    error: Exception,
    stage: str,
    msg: Any,
) -> None:
    payload = build_dead_letter_message(
        raw_value=raw_value,
        error=error,
        stage=stage,
        topic=msg.topic(),
        partition=msg.partition(),
        offset=msg.offset(),
        message_key=msg.key(),
    )
    producer.produce(TOPIC_DLQ, key=msg.key(), value=payload)
    producer.flush(5)
    MESSAGES_DLQ.inc()
    log_event(
        logger,
        logging.ERROR,
        "message_sent_to_dlq",
        dlq_topic=TOPIC_DLQ,
        error=str(error),
        offset=msg.offset(),
        partition=msg.partition(),
        source_topic=msg.topic(),
        stage=stage,
    )


class FeatureEngineer:
    def __init__(self, redis_client: Any):
        self.r = redis_client
        self.ttl_7d = 7 * 24 * 3600
        self.ttl_24h = 24 * 3600
        self.ttl_30d = 30 * 24 * 3600

    def _get_window_stats(self, zset_key: str, ts: float, window_sec: int) -> tuple[int, float]:
        cutoff = ts - window_sec
        members = self.r.zrangebyscore(zset_key, cutoff, ts)
        amounts = []
        for member in members:
            amount_text, _, _ = _as_text(member).partition(":")
            amounts.append(float(amount_text))
        return len(amounts), sum(amounts)

    def _get_user_stats(
        self,
        stats_key: str,
        amount: float,
    ) -> tuple[float, float, float, int, float, float, int]:
        raw_stats = self.r.hgetall(stats_key) or {}
        stats = {_as_text(k): _as_text(v) for k, v in raw_stats.items()}

        count_30d = int(float(stats.get("count_30d", 0)))
        sum_30d = float(stats.get("sum_30d", 0.0))
        sumsq_30d = float(stats.get("sumsq_30d", 0.0))
        fraud_count_30d = int(float(stats.get("fraud_count_30d", 0)))

        if count_30d == 0:
            mean_30d = amount
            std_30d = 1.0
        else:
            mean_30d = sum_30d / count_30d
            variance = max((sumsq_30d / count_30d) - (mean_30d ** 2), 0.0)
            std_30d = max(math.sqrt(variance), 1.0)

        user_chargeback_rate = (fraud_count_30d / count_30d) if count_30d else 0.0
        return mean_30d, std_30d, user_chargeback_rate, count_30d, sum_30d, sumsq_30d, fraud_count_30d

    def _update_user_stats(
        self,
        stats_key: str,
        amount: float,
        count_30d: int,
        sum_30d: float,
        sumsq_30d: float,
        fraud_count_30d: int,
        is_fraud: int,
    ) -> None:
        self.r.hset(
            stats_key,
            mapping={
                "count_30d": count_30d + 1,
                "sum_30d": round(sum_30d + amount, 6),
                "sumsq_30d": round(sumsq_30d + (amount ** 2), 6),
                "fraud_count_30d": fraud_count_30d + is_fraud,
            },
        )
        self.r.expire(stats_key, self.ttl_30d)

    def compute(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        user_id = raw["user_id"]
        tx_id = raw["transaction_id"]
        ts = raw["timestamp"]
        amount = raw["amount"]
        merchant_id = raw["merchant_id"]
        country = raw["country"]

        # Use the current transaction in rolling velocity windows.
        zset_key = f"user:{user_id}:txs"
        member = f"{amount}:{tx_id}"
        self.r.zadd(zset_key, {member: ts})
        cutoff_7d = ts - self.ttl_7d
        self.r.zremrangebyscore(zset_key, "-inf", cutoff_7d)
        self.r.expire(zset_key, self.ttl_7d)

        tx_count_1h, tx_sum_1h = self._get_window_stats(zset_key, ts, 3600)
        tx_count_24h, tx_sum_24h = self._get_window_stats(zset_key, ts, 86400)

        merchants_key = f"user:{user_id}:merchants:24h"
        countries_key = f"user:{user_id}:countries:7d"
        self.r.pfadd(merchants_key, merchant_id)
        self.r.expire(merchants_key, self.ttl_24h)
        unique_merchants_24h = self.r.pfcount(merchants_key)

        self.r.pfadd(countries_key, country)
        self.r.expire(countries_key, self.ttl_7d)
        unique_countries_7d = self.r.pfcount(countries_key)

        stats_key = f"user:{user_id}:stats"
        (
            mean_30d,
            std_30d,
            user_chargeback_rate,
            count_30d,
            sum_30d,
            sumsq_30d,
            fraud_count_30d,
        ) = self._get_user_stats(stats_key, amount)

        amount_zscore = (amount - mean_30d) / std_30d
        amount_log = math.log1p(amount)

        merchant_key = f"merchant:{merchant_id}:fraud_rate"
        merchant_fraud_rate_30d = float(self.r.get(merchant_key) or 0.0)

        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        day_of_week = dt.weekday()

        features = {
            "transaction_id": tx_id,
            "user_id": user_id,
            "timestamp": ts,
            "amount": amount,
            "amount_log": amount_log,
            "amount_zscore": amount_zscore,
            "tx_count_1h": tx_count_1h,
            "tx_count_24h": tx_count_24h,
            "tx_sum_1h": tx_sum_1h,
            "tx_sum_24h": tx_sum_24h,
            "unique_merchants_24h": int(unique_merchants_24h),
            "unique_countries_7d": int(unique_countries_7d),
            "hour_of_day": dt.hour,
            "day_of_week": day_of_week,
            "is_weekend": bool(day_of_week >= 5),
            "card_present": raw["card_present"],
            "merchant_category": raw["merchant_category"],
            "merchant_fraud_rate_30d": merchant_fraud_rate_30d,
            "user_chargeback_rate": user_chargeback_rate,
            "label": raw.get("_is_fraud"),
        }

        self._update_user_stats(
            stats_key=stats_key,
            amount=amount,
            count_30d=count_30d,
            sum_30d=sum_30d,
            sumsq_30d=sumsq_30d,
            fraud_count_30d=fraud_count_30d,
            is_fraud=int(raw.get("_is_fraud") or 0),
        )
        return features


def ensure_topic_exists(admin_client: Any, topic_name: str, num_partitions: int = 6) -> None:
    metadata = admin_client.list_topics(timeout=10)
    if topic_name in metadata.topics:
        return

    new_topic = NewTopic(topic_name, num_partitions=num_partitions, replication_factor=1)
    futures = admin_client.create_topics([new_topic])
    futures[topic_name].result()


def main():
    if redis is None or Consumer is None or Producer is None or AdminClient is None:
        raise RuntimeError(
            "consumer service requires redis and confluent-kafka packages. "
            "Install service dependencies before running it."
        )

    _start_metrics_server()
    r = redis.from_url(REDIS_URL, decode_responses=True)
    fe = FeatureEngineer(r)
    admin_client = AdminClient({"bootstrap.servers": KAFKA_BOOTSTRAP_SERVERS})
    ensure_topic_exists(admin_client, TOPIC_FEATURES)
    ensure_topic_exists(admin_client, TOPIC_DLQ, num_partitions=3)

    consumer_conf = {
        "bootstrap.servers": KAFKA_BOOTSTRAP_SERVERS,
        "group.id": KAFKA_GROUP_ID,
        "auto.offset.reset": "latest",
    }
    consumer = Consumer(consumer_conf)
    consumer.subscribe([TOPIC_TRANSACTIONS])

    producer_conf = {
        "bootstrap.servers": KAFKA_BOOTSTRAP_SERVERS,
        "linger.ms": 2,
        "compression.type": "snappy",
    }
    producer = Producer(producer_conf)

    log_event(
        logger,
        logging.INFO,
        "consumer_started",
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        dlq_topic=TOPIC_DLQ,
        features_topic=TOPIC_FEATURES,
        raw_topic=TOPIC_TRANSACTIONS,
    )

    try:
        while True:
            msg = consumer.poll(1.0)
            if msg is None:
                continue
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    continue

                MESSAGES_FAILED.labels(stage="consume").inc()
                log_event(logger, logging.ERROR, "consumer_error", error=str(msg.error()))
                break

            raw_value = msg.value()
            try:
                raw_tx = json.loads(raw_value.decode("utf-8"))
            except Exception as exc:
                MESSAGES_FAILED.labels(stage="deserialize").inc()
                publish_dead_letter(producer, raw_value, exc, "deserialize", msg)
                continue

            try:
                start_time = time.perf_counter()
                feature_vector = fe.compute(raw_tx)
                FEATURE_COMPUTE_LATENCY.observe(time.perf_counter() - start_time)
                producer.produce(
                    TOPIC_FEATURES,
                    key=feature_vector["user_id"],
                    value=json.dumps(feature_vector),
                )
                producer.poll(0)
                MESSAGES_PROCESSED.inc()
            except Exception as exc:
                MESSAGES_FAILED.labels(stage="transform").inc()
                publish_dead_letter(producer, raw_value, exc, "transform", msg)

    except KeyboardInterrupt:
        log_event(logger, logging.INFO, "consumer_shutdown_requested")
    finally:
        consumer.close()
        producer.flush()


if __name__ == "__main__":
    main()
