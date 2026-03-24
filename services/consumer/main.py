import json
import os
import time
import math
from datetime import datetime, timezone
from typing import Dict, Any

import redis
import numpy as np
from confluent_kafka import Consumer, Producer, KafkaError

# Environment Variables
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_GROUP_ID = os.getenv("KAFKA_GROUP_ID", "feature-engineer-group")
TOPIC_TRANSACTIONS = os.getenv("TOPIC_TRANSACTIONS", "transactions")
TOPIC_FEATURES = os.getenv("TOPIC_FEATURES", "features")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

class FeatureEngineer:
    def __init__(self, redis_client: redis.Redis):
        self.r = redis_client
        self.ttl_7d = 7 * 24 * 3600
        self.ttl_24h = 24 * 3600

    def compute(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        user_id = raw["user_id"]
        tx_id = raw["transaction_id"]
        ts = raw["timestamp"]
        amount = raw["amount"]
        merchant_id = raw["merchant_id"]
        country = raw["country"]

        # 1. Velocity Features (Sorted Sets)
        zset_key = f"user:{user_id}:txs"
        member = f"{amount}:{tx_id}"
        
        # Add current transaction
        self.r.zadd(zset_key, {member: ts})
        
        # Trim old data (> 7 days) and set TTL
        cutoff_7d = ts - self.ttl_7d
        self.r.zremrangebyscore(zset_key, "-inf", cutoff_7d)
        self.r.expire(zset_key, self.ttl_7d)

        # Query ranges
        def get_stats(window_sec):
            cutoff = ts - window_sec
            members = self.r.zrangebyscore(zset_key, cutoff, ts)
            count = len(members)
            total_sum = sum(float(m.decode().split(":")[0]) for m in members)
            return count, total_sum

        tx_count_1h, tx_sum_1h = get_stats(3600)
        tx_count_24h, tx_sum_24h = get_stats(86400)

        # 2. Cardinality (HyperLogLog)
        merchants_key = f"user:{user_id}:merchants:24h"
        countries_key = f"user:{user_id}:countries:7d"
        
        self.r.pfadd(merchants_key, merchant_id)
        self.r.expire(merchants_key, 86400)
        unique_merchants_24h = self.r.pfcount(merchants_key)

        self.r.pfadd(countries_key, country)
        self.r.expire(countries_key, self.ttl_7d)
        unique_countries_7d = self.r.pfcount(countries_key)

        # 3. Stats & Z-Score
        stats_key = f"user:{user_id}:stats"
        user_stats = self.r.hgetall(stats_key)
        
        mean_30d = float(user_stats.get(b"mean_30d", amount))
        std_30d = float(user_stats.get(b"std_30d", 1.0))
        user_chargeback_rate = float(user_stats.get(b"user_chargeback_rate", 0.0))
        
        amount_zscore = (amount - mean_30d) / std_30d
        amount_log = math.log(amount + 1.0)

        # 4. Merchant Fraud Rate
        merchant_key = f"merchant:{merchant_id}:fraud_rate"
        merchant_fraud_rate_30d = float(self.r.get(merchant_key) or 0.0)

        # 5. Temporal Features
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        hour_of_day = dt.hour
        day_of_week = dt.weekday()
        is_weekend = day_of_week >= 5

        # Feature Vector
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
            "unique_merchants_24h": unique_merchants_24h,
            "unique_countries_7d": int(unique_countries_7d),
            "hour_of_day": hour_of_day,
            "day_of_week": day_of_week,
            "is_weekend": bool(is_weekend),
            "card_present": raw["card_present"],
            "merchant_category": raw["merchant_category"],
            "merchant_fraud_rate_30d": merchant_fraud_rate_30d,
            "user_chargeback_rate": user_chargeback_rate,
            "label": raw.get("_is_fraud")
        }
        return features

def main():
    # Redis init
    r = redis.from_url(REDIS_URL)
    fe = FeatureEngineer(r)

    # Kafka Consumer config
    consumer_conf = {
        "bootstrap.servers": KAFKA_BOOTSTRAP_SERVERS,
        "group.id": KAFKA_GROUP_ID,
        "auto.offset.reset": "latest"
    }
    consumer = Consumer(consumer_conf)
    consumer.subscribe([TOPIC_TRANSACTIONS])

    # Kafka Producer config
    producer_conf = {
        "bootstrap.servers": KAFKA_BOOTSTRAP_SERVERS,
        "linger.ms": 2,
        "compression.type": "snappy"
    }
    producer = Producer(producer_conf)

    print(f"Consumer started. Subscribed to {TOPIC_TRANSACTIONS}...")

    try:
        while True:
            msg = consumer.poll(1.0)
            if msg is None:
                continue
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    continue
                else:
                    print(f"Consumer error: {msg.error()}")
                    break

            # Process message
            try:
                raw_tx = json.loads(msg.value().decode("utf-8"))
                feature_vector = fe.compute(raw_tx)
                
                # Publish features
                producer.produce(
                    TOPIC_FEATURES,
                    key=feature_vector["user_id"],
                    value=json.dumps(feature_vector)
                )
                producer.poll(0)
            except Exception as e:
                print(f"Error processing message: {e}")

    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        consumer.close()
        producer.flush()

if __name__ == "__main__":
    main()
