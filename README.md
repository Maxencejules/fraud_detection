# Real-Time Fraud Detection Pipeline

An end-to-end fraud detection project built as a streaming system: synthetic card transactions are published to Kafka, enriched into real-time behavioral features, scored by a FastAPI inference service backed by an MLflow-registered model, and monitored for drift with Evidently.

## What This Project Demonstrates

- Event-driven architecture with Kafka, Redis, FastAPI, MLflow, PostgreSQL, and Docker Compose
- Real-time feature engineering for transaction velocity, merchant diversity, geographic spread, and user risk history
- Offline model training with XGBoost and LightGBM, experiment tracking, and automatic model promotion
- Production-style monitoring using cached predictions plus drift and data-quality reports
- Lightweight automated tests for feature engineering and inference decisioning

## Architecture

`producer` -> publishes synthetic raw transactions to `transactions.raw`

`consumer` -> computes rolling fraud features and publishes to `transactions.features`

`predictor` -> loads the champion model from MLflow and serves `/predict`

`monitor` -> consumes engineered features, joins cached predictions from Redis, and logs Evidently reports to MLflow

`trainer` -> trains candidate models from parquet feature data and promotes the best registry version

## Project Layout

```text
services/
  producer/   synthetic transaction generator
  consumer/   streaming feature engineering
  predictor/  FastAPI inference API
  trainer/    offline training and MLflow registration
  monitor/    drift and quality reporting
scripts/
  gen_training_data.py
  benchmark_latency.py
tests/
  test_feature_engineer.py
  test_predictor.py
```

## Quick Start

1. Generate local training and reference data:

```powershell
python scripts/gen_training_data.py
```

2. Start infrastructure and streaming services:

```powershell
docker compose up --build
```

3. Train and register a model:

```powershell
docker compose --profile train run --rm trainer
```

4. Exercise the predictor:

```powershell
python scripts/benchmark_latency.py --url http://localhost:8000 --n 200 --concurrency 10
```

## Local Test Run

Install the lightweight test dependencies:

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

The tests are intentionally lightweight: they validate core feature engineering behavior and inference thresholding without requiring Kafka, Redis, or MLflow to be running locally.

## Implementation Notes

- The predictor caches recent decisions in Redis under `pred:<transaction_id>` so the monitoring service can enrich drift reports with model outputs.
- The consumer maintains rolling counts and sums over Redis sorted sets and approximate cardinality over HyperLogLog keys.
- User-level running statistics are stored in Redis hashes to support amount z-scores and historical chargeback rate features.
- Dockerfiles are service-specific and assume the repo root as the Docker build context.

## Resume Positioning

Possible resume bullet:

Built a real-time fraud detection platform using Kafka, Redis, FastAPI, MLflow, and Docker; engineered streaming behavioral features, trained and promoted gradient-boosted models, and added drift monitoring plus automated API and feature tests.
