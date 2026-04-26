# Real-Time Fraud Detection Pipeline

Production-style fraud detection demo that combines streaming feature engineering with synchronous model serving.

Raw card transactions flow through Kafka, user and merchant behavior is aggregated in Redis, a FastAPI predictor serves fraud decisions from an MLflow-registered model, and an Evidently-based monitor publishes drift reports to MLflow artifacts.

## Verified Run

Artifacts were generated locally on April 25, 2026.

- Training quality from `output/runtime/artifact_summary.json`: `xgb PR-AUC 1.0000`, `lightgbm PR-AUC 0.9998`, `ensemble PR-AUC 0.99998`.
- Predictor-reported serving latency from `output/runtime/benchmark_latest.json`: `mean 10.92 ms`, `p50 9.29 ms`, `p95 19.66 ms`.
- End-to-end client timing in the same local fallback run: `p50 4110.92 ms`, `p95 4133.96 ms`, `p99 4135.80 ms`, `0/60` errors at concurrency `5`.
- Evidently drift report generated at `reports/local_drift_report_1777095421.html` with `drift_share 0.30`.
- Supporting screenshots were captured to `output/screenshots/mlflow_training.png`, `output/screenshots/mlflow_monitoring.png`, and `output/screenshots/evidently_report.png`.

The benchmark caveat matters: the low double-digit millisecond numbers reflect predictor-side inference latency, while the multi-second client timings came from a local non-Docker fallback path after Docker Desktop failed on this machine with a WSL backend error. If you rerun the Compose flow on a healthy Docker host, publish those end-to-end numbers instead.

## Why This Repo Signals Well

- It is not just a notebook plus a classifier. The repo includes Kafka, Redis, FastAPI, MLflow, PostgreSQL, Docker Compose, and automated tests.
- The system has operational safeguards: a dead-letter topic for bad events, Prometheus metrics, health and readiness endpoints, and drift monitoring.
- The repo includes both fast unit tests and a full Docker smoke path in GitHub Actions.
- There is a benchmark harness and a smoke script so you can produce artifacts instead of making vague performance claims.

## Architecture

`producer` -> publishes synthetic raw transactions to `transactions.raw`

`consumer` -> computes rolling fraud features, emits feature events to `transactions.features`, and routes malformed events to `transactions.raw.dlq`

`predictor` -> serves `/predict`, `/health`, `/ready`, and `/metrics` using the champion model from MLflow

`monitor` -> consumes engineered features, joins cached predictions when present, and logs Evidently drift reports to MLflow

`trainer` -> trains XGBoost and LightGBM candidates, logs evaluation metrics, and promotes the best registry version

## Production-Minded Details

- Rolling features include transaction velocity, spend accumulation, merchant diversity, country spread, merchant fraud rate, and user chargeback rate.
- The consumer now emits dead-letter events with source topic, partition, offset, and original payload context.
- The predictor exposes Prometheus metrics and a readiness gate, which makes CI and deployment checks much cleaner.
- The monitor publishes drift reports as HTML artifacts and surfaces its own runtime counters for report generation and cache misses.
- CI validates both isolated logic and a real Docker stack that trains a model, reloads it, starts traffic, and waits for a drift report.

## Project Layout

```text
services/
  producer/   synthetic transaction generator
  consumer/   streaming feature engineering + DLQ handling
  predictor/  FastAPI fraud scoring API + Prometheus metrics
  trainer/    offline training + MLflow model registration
  monitor/    Evidently drift and data-quality reporting
scripts/
  gen_training_data.py
  benchmark_latency.py
  smoke_test_stack.py
tests/
  test_feature_engineer.py
  test_predictor.py
```

## Demo Flow

1. Install local helper dependencies for tests and scripts:

```powershell
python -m pip install -r requirements-dev.txt
```

2. Generate training and reference data:

```powershell
python scripts/gen_training_data.py
```

3. Start infrastructure and long-running services:

```powershell
docker compose up --build zookeeper kafka redis postgres mlflow consumer predictor monitor
```

4. Train and register the first production model:

```powershell
docker compose --profile train run --rm trainer
```

5. Reload the predictor so it serves the newly promoted model:

```powershell
Invoke-RestMethod -Method Post http://localhost:8000/reload-model
```

6. Start live traffic:

```powershell
docker compose up producer
```

7. Run the smoke test:

```powershell
python scripts/smoke_test_stack.py --base-url http://localhost:8000 --reports-dir reports --timeout 180
```

8. Capture a benchmark artifact for your README, resume, or LinkedIn post:

```powershell
python scripts/benchmark_latency.py --url http://localhost:8000 --n 500 --concurrency 20 --json-out reports/benchmark_latest.json
```

## Local Test Run

```powershell
python -m pytest -q
```

The unit tests are intentionally fast. They validate feature engineering behavior, DLQ payload construction, inference decisions, readiness behavior, and the predictor metrics endpoint without requiring Kafka, Redis, or MLflow to be running.

## What To Show Recruiters

- A short GIF or screenshot sequence of Kafka UI, MLflow runs, `/metrics`, and an Evidently report in `reports/`.
- A benchmark JSON file plus a one-line summary such as `P99 latency under X ms at Y concurrent requests`.
- The GitHub Actions run that proves the repo is tested and the stack actually boots.
- A concise project summary:

`Built a real-time fraud detection system with Kafka, Redis, FastAPI, MLflow, and Evidently, including online feature engineering, automated model promotion, Prometheus metrics, DLQ handling, and CI-backed Docker smoke tests.`

For this repo's current evidence set, a defensible version is:

`Built a real-time fraud detection system with Kafka, Redis, FastAPI, MLflow, and Evidently, with ensemble PR-AUC of 0.99998, predictor-side p95 latency of 19.66 ms, drift reporting, DLQ handling, Prometheus metrics, and CI-backed smoke coverage.`

## Useful Endpoints

- `GET /health` returns process health and model load state
- `GET /ready` returns `200` only when a serving model is loaded
- `GET /metrics` exposes Prometheus counters and latency histograms
- `POST /predict` scores one engineered transaction payload
- `POST /reload-model` refreshes the serving model from MLflow

## Current Boundary

This repo streams real-time feature generation and exposes fraud scoring through a low-latency API. The next obvious extension is a dedicated scoring consumer that reads `transactions.features` and emits a `transactions.decisions` topic for fully automated stream-to-decision processing.
