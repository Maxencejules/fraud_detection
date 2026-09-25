"""Latency/throughput benchmark for the predictor, plus the streaming end-to-end latency.

Payloads are realistic: feature vectors produced by the project's own simulator and
feature pipeline (not uniform random numbers). Reports client-side latency (includes
HTTP and client overhead), server-side latency (``X-Latency-Ms``), throughput and
errors, and, when the scorer is running, end-to-end percentiles estimated from its
Prometheus histogram. Results go to stdout and optionally to a JSON file.

Usage:
    uv run python scripts/benchmark.py --requests 2000 --concurrency 8 --json-out bench.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import platform
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import fakeredis
import httpx
import numpy as np

from fraud_detection.bootstrap import replay
from fraud_detection.features import FEATURE_COLUMNS, FeatureEngineer
from fraud_detection.simulation import DAY, SimulationConfig, TransactionSimulator


def realistic_payloads(count: int, seed: int = 7) -> list[dict[str, Any]]:
    config = SimulationConfig(seed=seed, n_users=500, n_merchants=120)
    end = time.time()
    frame = replay(
        FeatureEngineer(fakeredis.FakeRedis()),
        TransactionSimulator(config, stream="benchmark").events(end - 5 * DAY, end),
        batch_size=500,
    )
    sample = frame.sample(n=count, replace=len(frame) < count, random_state=seed)
    payloads = []
    for i, row in enumerate(sample[list(FEATURE_COLUMNS)].to_dict("records")):
        features = {str(k): v.item() if hasattr(v, "item") else v for k, v in row.items()}
        payloads.append({"transaction_id": f"bench-{i}", **features})
    return payloads


def percentiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    array = np.asarray(values)
    return {
        "p50": round(float(np.percentile(array, 50)), 3),
        "p95": round(float(np.percentile(array, 95)), 3),
        "p99": round(float(np.percentile(array, 99)), 3),
        "max": round(float(array.max()), 3),
        "mean": round(float(array.mean()), 3),
    }


async def hammer(
    base_url: str, payloads: list[dict[str, Any]], concurrency: int, batch_size: int
) -> dict[str, Any]:
    client_ms: list[float] = []
    server_ms: list[float] = []
    errors = 0
    if batch_size > 1:
        requests = [
            ("/v1/predict/batch", {"items": payloads[i : i + batch_size]})
            for i in range(0, len(payloads), batch_size)
        ]
    else:
        requests = [("/v1/predict", payload) for payload in payloads]
    queue: asyncio.Queue[tuple[str, dict[str, Any]]] = asyncio.Queue()
    for item in requests:
        queue.put_nowait(item)

    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    async with httpx.AsyncClient(base_url=base_url, timeout=10.0, limits=limits) as client:

        async def worker() -> None:
            nonlocal errors
            while not queue.empty():
                path, body = queue.get_nowait()
                started = time.perf_counter()
                try:
                    response = await client.post(path, json=body)
                except httpx.HTTPError:
                    errors += 1
                    continue
                elapsed = (time.perf_counter() - started) * 1000
                if response.status_code != 200:
                    errors += 1
                    continue
                client_ms.append(elapsed)
                if "X-Latency-Ms" in response.headers:
                    server_ms.append(float(response.headers["X-Latency-Ms"]))
                else:
                    server_ms.append(float(response.json()["latency_ms"]))

        started = time.perf_counter()
        await asyncio.gather(*(worker() for _ in range(concurrency)))
        wall = time.perf_counter() - started

    transactions = len(client_ms) * batch_size
    return {
        "requests": len(requests),
        "errors": errors,
        "wall_seconds": round(wall, 3),
        "throughput_rps": round(len(client_ms) / wall, 1),
        "throughput_tps": round(transactions / wall, 1),
        "client_latency_ms": percentiles(client_ms),
        "server_latency_ms": percentiles(server_ms),
    }


Buckets = list[tuple[float, float]]


def scrape_histogram(metrics_url: str, metric: str) -> Buckets:
    """Cumulative (upper bound, count) pairs of a Prometheus histogram."""
    try:
        text = httpx.get(metrics_url, timeout=5).text
    except httpx.HTTPError:
        return []
    pattern = rf'^{metric}_bucket\{{le="([^"]+)"\}} ([0-9.eE+-]+)$'
    buckets = [
        (math.inf if bound == "+Inf" else float(bound), float(count))
        for bound, count in re.findall(pattern, text, re.MULTILINE)
    ]
    return sorted(buckets)


def histogram_quantiles(buckets: Buckets, baseline: Buckets | None = None) -> dict[str, float]:
    """Estimate quantiles (ms), optionally over the window since ``baseline``.

    Linear interpolation within buckets, like PromQL's ``histogram_quantile``.
    """
    if baseline:
        before = dict(baseline)
        buckets = [(bound, count - before.get(bound, 0.0)) for bound, count in buckets]
    total = buckets[-1][1] if buckets else 0.0
    if total <= 0:
        return {}
    result: dict[str, float] = {"count": total}
    for name, q in (("p50", 0.5), ("p95", 0.95), ("p99", 0.99)):
        rank, previous_bound, previous_count = q * total, 0.0, 0.0
        for bound, count in buckets:
            if count >= rank:
                if math.isinf(bound):
                    estimate = previous_bound
                else:
                    share = (rank - previous_count) / max(count - previous_count, 1e-9)
                    estimate = previous_bound + (bound - previous_bound) * share
                result[name] = round(estimate * 1000, 1)
                break
            previous_bound, previous_count = bound, count
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark the fraud predictor.")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--requests", type=int, default=2_000)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1, help=">1 uses /v1/predict/batch")
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--scorer-metrics", default="http://localhost:9110/metrics")
    parser.add_argument(
        "--stream-window-s",
        type=float,
        default=60.0,
        help="measure steady-state streaming latency over this window first (0 to skip)",
    )
    parser.add_argument("--max-p99-ms", type=float, default=None, help="fail above this")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    model = httpx.get(f"{args.url}/v1/model", timeout=10).json()
    stream: dict[str, float] = {}
    if args.stream_window_s > 0:
        metric = "fraud_scorer_end_to_end_seconds"
        baseline = scrape_histogram(args.scorer_metrics, metric)
        time.sleep(args.stream_window_s)
        stream = histogram_quantiles(scrape_histogram(args.scorer_metrics, metric), baseline)
    payloads = realistic_payloads(args.warmup + args.requests * args.batch_size)
    asyncio.run(hammer(args.url, payloads[: args.warmup], args.concurrency, 1))  # warm-up
    result = asyncio.run(
        hammer(args.url, payloads[args.warmup :], args.concurrency, args.batch_size)
    )
    report = {
        "timestamp": datetime.now(tz=UTC).isoformat(timespec="seconds"),
        "target": args.url,
        "endpoint": "/v1/predict/batch" if args.batch_size > 1 else "/v1/predict",
        "batch_size": args.batch_size,
        "concurrency": args.concurrency,
        "model_version": model.get("version"),
        **result,
        "stream_end_to_end_ms": {"window_s": args.stream_window_s, **stream},
        "environment": {
            "cpus": os.cpu_count(),
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
    }
    print(json.dumps(report, indent=2))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    p99 = report["client_latency_ms"].get("p99", math.inf)
    if result["errors"] or (args.max_p99_ms is not None and p99 > args.max_p99_ms):
        print(f"FAIL: errors={result['errors']} p99={p99}ms", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
