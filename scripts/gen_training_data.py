import os
import uuid
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Constants
TRAIN_ROWS = 203_000
REF_ROWS = 5_000
FRAUD_RATE = 0.015

FEATURE_COLS = [
    "amount", "amount_log", "amount_zscore", "tx_count_1h", "tx_count_24h",
    "tx_sum_1h", "tx_sum_24h", "unique_merchants_24h", "unique_countries_7d",
    "hour_of_day", "day_of_week", "is_weekend", "card_present",
    "merchant_fraud_rate_30d", "user_chargeback_rate"
]
TARGET_COL = "label"

def generate_batch(n, is_fraud=False):
    data = {}
    
    if not is_fraud:
        # Legit
        data["amount"] = np.random.lognormal(3.5, 1.2, n).clip(min=0.01)
        data["amount_zscore"] = np.random.normal(0, 1, n)
        data["tx_count_1h"] = np.random.poisson(2, n)
        data["tx_count_24h"] = np.random.poisson(8, n)
        data["tx_sum_1h"] = np.random.exponential(50, n)
        data["tx_sum_24h"] = np.random.exponential(200, n)
        data["unique_merchants_24h"] = np.random.randint(1, 6, n)
        data["unique_countries_7d"] = np.ones(n, dtype=int)
        
        # Weighted hours: 9-17 (0.7 weight), others (0.3 weight)
        business_hours = np.random.randint(9, 18, int(n * 0.7))
        other_hours = np.random.randint(0, 24, n - len(business_hours))
        data["hour_of_day"] = np.concatenate([business_hours, other_hours])
        np.random.shuffle(data["hour_of_day"])
        
        data["card_present"] = np.random.random(n) < 0.7
        data["merchant_fraud_rate_30d"] = np.random.beta(1, 50, n)
        data["user_chargeback_rate"] = np.random.beta(1, 100, n)
        data[TARGET_COL] = np.zeros(n, dtype=int)
    else:
        # Fraud
        # 50/50 Large or Micro
        mask = np.random.random(n) < 0.5
        amounts = np.zeros(n)
        amounts[mask] = np.random.uniform(500, 5000, mask.sum())
        amounts[~mask] = np.random.uniform(0.01, 1.0, (~mask).sum())
        data["amount"] = amounts
        
        data["amount_zscore"] = np.random.normal(3.5, 1.5, n)
        data["tx_count_1h"] = np.random.poisson(8, n)
        data["tx_count_24h"] = np.random.poisson(25, n)
        data["tx_sum_1h"] = np.random.exponential(800, n)
        data["tx_sum_24h"] = np.random.exponential(3000, n)
        data["unique_merchants_24h"] = np.random.randint(4, 12, n)
        data["unique_countries_7d"] = np.random.randint(2, 5, n)
        data["hour_of_day"] = np.random.randint(0, 24, n)
        data["card_present"] = np.zeros(n, dtype=bool)
        data["merchant_fraud_rate_30d"] = np.random.beta(5, 10, n)
        data["user_chargeback_rate"] = np.random.beta(5, 50, n)
        data[TARGET_COL] = np.ones(n, dtype=int)

    # Common fields
    data["amount_log"] = np.log1p(data["amount"])
    data["day_of_week"] = np.random.randint(0, 7, n)
    data["is_weekend"] = data["day_of_week"] >= 5
    data["transaction_id"] = [str(uuid.uuid4()) for _ in range(n)]
    data["user_id"] = [f"user_{np.random.randint(1000, 9999)}" for _ in range(n)]
    data["timestamp"] = [datetime.now().timestamp() - np.random.randint(0, 86400*30) for _ in range(n)]

    return pd.DataFrame(data)

def main():
    os.makedirs("data", exist_ok=True)
    
    print("Generating training data...")
    num_fraud = int(TRAIN_ROWS * FRAUD_RATE)
    num_legit = TRAIN_ROWS - num_fraud
    
    df_legit = generate_batch(num_legit, is_fraud=False)
    df_fraud = generate_batch(num_fraud, is_fraud=True)
    
    df_train = pd.concat([df_legit, df_fraud], ignore_index=True)
    df_train = df_train.sample(frac=1).reset_index(drop=True)
    
    train_path = "data/features_train.parquet"
    df_train.to_parquet(train_path, engine="pyarrow")
    
    print("Generating reference data...")
    df_ref = generate_batch(REF_ROWS, is_fraud=False)
    # Ensure all required monitor columns exist
    df_ref["prediction"] = df_ref["label"].astype(float) # Dummy for initial state
    
    ref_path = "data/reference.parquet"
    df_ref.to_parquet(ref_path, engine="pyarrow")
    
    print("-" * 30)
    print(f"Training data: {len(df_train)} rows, {df_train[TARGET_COL].sum()} fraud ({df_train[TARGET_COL].mean():.2%})")
    print(f"Reference data: {len(df_ref)} rows (legit only)")
    print(f"Saved to {train_path} and {ref_path}")

if __name__ == "__main__":
    main()
