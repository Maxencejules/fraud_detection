import json
import os
import random
import time
import uuid

import numpy as np

try:
    from confluent_kafka import Producer
    from confluent_kafka.admin import AdminClient, NewTopic
except ImportError:  # pragma: no cover - runtime dependency only
    Producer = AdminClient = NewTopic = None

KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
TOPIC_TRANSACTIONS = os.getenv("TOPIC_TRANSACTIONS", "transactions.raw")
EMIT_RATE_TPS = float(os.getenv("EMIT_RATE_TPS", "10"))
FRAUD_RATE = float(os.getenv("FRAUD_RATE", "0.015"))

CATEGORIES = ["retail", "grocery", "food", "travel", "health", "entertainment", "electronics", "financial"]
COUNTRIES_LEGIT = ["US", "US", "US", "US", "CA", "GB", "DE", "FR"]
COUNTRIES_FRAUD = ["NG", "RU", "CN", "BR"]


def generate_transaction():
    is_fraud = random.random() < FRAUD_RATE

    if is_fraud:
        amount = random.uniform(500, 5000) if random.random() < 0.5 else random.uniform(0.01, 1.0)
        country = random.choice(COUNTRIES_FRAUD)
        card_present = False
    else:
        amount = float(np.random.lognormal(mean=3.5, sigma=1.2))
        country = random.choice(COUNTRIES_LEGIT)
        card_present = random.random() < 0.65

    return {
        "transaction_id": str(uuid.uuid4()),
        "user_id": f"user_{random.randint(1000, 9999)}",
        "merchant_id": f"merchant_{random.randint(100, 999)}",
        "amount": round(amount, 2),
        "currency": "USD",
        "timestamp": time.time(),
        "merchant_category": random.choice(CATEGORIES),
        "country": country,
        "card_present": card_present,
        "device_fingerprint": hex(random.getrandbits(64))[2:],
        "_is_fraud": int(is_fraud),
    }


def create_topic_if_not_exists(admin_client, topic_name):
    metadata = admin_client.list_topics(timeout=10)
    if topic_name in metadata.topics:
        return

    print(f"Creating topic {topic_name} with 6 partitions...")
    new_topic = NewTopic(topic_name, num_partitions=6, replication_factor=1)
    futures = admin_client.create_topics([new_topic])
    futures[topic_name].result()
    print(f"Topic {topic_name} created.")


def delivery_report(err, msg):
    if err is not None:
        print(f"Message delivery failed: {err}")


def main():
    if Producer is None or AdminClient is None:
        raise RuntimeError(
            "producer service requires confluent-kafka. Install service dependencies before running it."
        )

    conf = {
        "bootstrap.servers": KAFKA_BOOTSTRAP_SERVERS,
        "linger.ms": 5,
        "batch.num.messages": 500,
        "compression.type": "snappy",
    }

    admin_client = AdminClient({"bootstrap.servers": KAFKA_BOOTSTRAP_SERVERS})
    create_topic_if_not_exists(admin_client, TOPIC_TRANSACTIONS)

    producer = Producer(conf)
    print(f"Starting production at {EMIT_RATE_TPS} TPS...")

    delay = 1.0 / EMIT_RATE_TPS
    last_emit = time.monotonic()

    try:
        while True:
            tx = generate_transaction()
            producer.produce(
                TOPIC_TRANSACTIONS,
                key=tx["user_id"],
                value=json.dumps(tx),
                on_delivery=delivery_report,
            )

            elapsed = time.monotonic() - last_emit
            sleep_time = delay - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

            last_emit = time.monotonic()
            producer.poll(0)

    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        producer.flush()


if __name__ == "__main__":
    main()
