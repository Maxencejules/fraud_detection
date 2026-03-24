import os
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, brier_score_loss
)
import xgboost as xgb
import lightgbm as lgb
import mlflow
import mlflow.xgboost
import mlflow.lightgbm
from mlflow.tracking import MlflowClient

# Constants
FEATURE_COLS = [
    "amount", "amount_log", "amount_zscore", "tx_count_1h", "tx_count_24h",
    "tx_sum_1h", "tx_sum_24h", "unique_merchants_24h", "unique_countries_7d",
    "hour_of_day", "day_of_week", "is_weekend", "card_present",
    "merchant_fraud_rate_30d", "user_chargeback_rate"
]
TARGET_COL = "label"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
DATA_DIR = os.getenv("DATA_DIR", "/app/data")
MODEL_NAME = "fraud-detector"

def load_data():
    files = glob.glob(os.path.join(DATA_DIR, "features_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {DATA_DIR}")
    
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    fraud_rate = df[TARGET_COL].mean()
    print(f"Loaded {len(df)} rows. Fraud rate: {fraud_rate:.4%}")
    return df, fraud_rate

def evaluate(y_test, y_prob, prefix=""):
    y_pred = (y_prob > 0.5).astype(int)
    metrics = {
        f"{prefix}roc_auc": roc_auc_score(y_test, y_prob),
        f"{prefix}pr_auc": average_precision_score(y_test, y_prob),
        f"{prefix}f1": f1_score(y_test, y_pred),
        f"{prefix}precision": precision_score(y_test, y_pred),
        f"{prefix}recall": recall_score(y_test, y_pred),
        f"{prefix}brier_score_loss": brier_score_loss(y_test, y_prob)
    }
    return metrics

def train():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("fraud-detection-training")

    df, fraud_rate = load_data()
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    # Split: 80/10/10
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

    spw = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"Scale Pos Weight: {spw:.2f}")

    with mlflow.start_run() as run:
        # Log params
        mlflow.log_params({
            "train_size": len(X_train),
            "val_size": len(X_val),
            "test_size": len(X_test),
            "fraud_rate": fraud_rate,
            "scale_pos_weight": spw,
            "features": FEATURE_COLS
        })

        # XGBoost
        xgb_model = xgb.XGBClassifier(
            n_estimators=800,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=10,
            scale_pos_weight=spw,
            objective="binary:logistic",
            eval_metric="aucpr",
            tree_method="hist",
            early_stopping_rounds=50,
            random_state=42
        )
        xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        xgb_prob = xgb_model.predict_proba(X_test)[:, 1]
        xgb_metrics = evaluate(y_test, xgb_prob, prefix="xgb_")
        mlflow.log_metrics(xgb_metrics)
        mlflow.xgboost.log_model(xgb_model, "xgb_model")

        # LightGBM
        lgb_model = lgb.LGBMClassifier(
            n_estimators=800,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=10,
            scale_pos_weight=spw,
            objective="binary",
            metric="aucpr",
            random_state=42,
            n_jobs=-1
        )
        lgb_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50)]
        )
        lgb_prob = lgb_model.predict_proba(X_test)[:, 1]
        lgb_metrics = evaluate(y_test, lgb_prob, prefix="lgb_")
        mlflow.log_metrics(lgb_metrics)
        
        # Register LightGBM as "fraud-detector"
        mlflow.lightgbm.log_model(
            lgb_model, 
            "lgb_model", 
            registered_model_name=MODEL_NAME
        )

        # Ensemble (soft vote)
        ensemble_prob = (xgb_prob + lgb_prob) / 2
        ensemble_metrics = evaluate(y_test, ensemble_prob, prefix="ensemble_")
        mlflow.log_metrics(ensemble_metrics)

        # Promotion Logic
        client = MlflowClient()
        new_pr_auc = ensemble_metrics["ensemble_pr_auc"]
        promote = False
        
        try:
            prod_versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])
            if not prod_versions:
                promote = True
            else:
                prod_run_id = prod_versions[0].run_id
                prod_metrics = client.get_run(prod_run_id).data.metrics
                current_pr_auc = prod_metrics.get("ensemble_pr_auc", 0.0)
                
                print(f"New PR_AUC: {new_pr_auc:.4f}, Current Production PR_AUC: {current_pr_auc:.4f}")
                if new_pr_auc > current_pr_auc:
                    promote = True
        except Exception as e:
            print(f"Error checking production versions: {e}")
            promote = True

        # Latest version is the one we just registered
        latest_version = client.get_latest_versions(MODEL_NAME, stages=["None"])[0].version
        
        if promote:
            print(f"Promoting version {latest_version} to Production")
            client.transition_model_version_stage(
                name=MODEL_NAME,
                version=latest_version,
                stage="Production",
                archive_existing_versions=True
            )
        else:
            print(f"Archiving version {latest_version}")
            client.transition_model_version_stage(
                name=MODEL_NAME,
                version=latest_version,
                stage="Archived"
            )

if __name__ == "__main__":
    train()
