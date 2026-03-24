import os

# Feature Definitions
FEATURE_COLS = [
    "amount", "amount_log", "amount_zscore", "tx_count_1h", "tx_count_24h",
    "tx_sum_1h", "tx_sum_24h", "unique_merchants_24h", "unique_countries_7d",
    "hour_of_day", "day_of_week", "is_weekend", "card_present",
    "merchant_fraud_rate_30d", "user_chargeback_rate"
]

CATEGORICAL_FEATURES = ["is_weekend", "card_present", "day_of_week"]
NUMERICAL_FEATURES = [c for c in FEATURE_COLS if c not in CATEGORICAL_FEATURES]

# Kafka Topics
TOPIC_RAW = "transactions.raw"
TOPIC_FEATURES = "transactions.features"

# Monitor Configuration
CONSUME_TOPIC = TOPIC_FEATURES
MONITOR_EXPERIMENT_NAME = "fraud-monitoring"
MODEL_NAME = "fraud-detector"
