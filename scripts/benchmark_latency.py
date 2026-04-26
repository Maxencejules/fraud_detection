import argparse
import json
import time
import random
import uuid
import requests
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any

FEATURE_COLS = [
    "amount", "amount_log", "amount_zscore", "tx_count_1h", "tx_count_24h",
    "tx_sum_1h", "tx_sum_24h", "unique_merchants_24h", "unique_countries_7d",
    "hour_of_day", "day_of_week", "is_weekend", "card_present",
    "merchant_fraud_rate_30d", "user_chargeback_rate"
]

def random_payload() -> Dict[str, Any]:
    amount = random.uniform(1.0, 1000.0)
    return {
        "transaction_id": str(uuid.uuid4()),
        "amount": amount,
        "amount_log": float(np.log1p(amount)),
        "amount_zscore": random.uniform(-1.0, 1.0),
        "tx_count_1h": random.randint(0, 5),
        "tx_count_24h": random.randint(0, 20),
        "tx_sum_1h": random.uniform(0.0, 500.0),
        "tx_sum_24h": random.uniform(0.0, 2000.0),
        "unique_merchants_24h": random.randint(1, 5),
        "unique_countries_7d": random.randint(1, 2),
        "hour_of_day": random.randint(0, 23),
        "day_of_week": random.randint(0, 6),
        "is_weekend": random.choice([True, False]),
        "card_present": random.choice([True, False]),
        "merchant_fraud_rate_30d": random.uniform(0.0, 0.05),
        "user_chargeback_rate": random.uniform(0.0, 0.01)
    }

def fire_request(url: str):
    payload = random_payload()
    start_time = time.perf_counter()
    try:
        response = requests.post(f"{url}/predict", json=payload, timeout=2.0)
        latency = (time.perf_counter() - start_time) * 1000
        if response.status_code == 200:
            return latency, True
        return latency, False
    except Exception:
        latency = (time.perf_counter() - start_time) * 1000
        return latency, False

def run_benchmark(url: str, n: int, concurrency: int):
    print(f"Benchmarking {url} with {n} requests (concurrency={concurrency})...")
    
    latencies = []
    success_count = 0
    error_count = 0
    
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(fire_request, url) for _ in range(n)]
        for future in futures:
            latency, success = future.result()
            latencies.append(latency)
            if success:
                success_count += 1
            else:
                error_count += 1
                
    latencies = np.array(latencies)
    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)
    p99 = np.percentile(latencies, 99)
    p999 = np.percentile(latencies, 99.9)
    mean = np.mean(latencies)
    max_lat = np.max(latencies)
    
    result = {
        "requests_ok": success_count,
        "errors": error_count,
        "mean_ms": round(float(mean), 2),
        "p50_ms": round(float(p50), 2),
        "p95_ms": round(float(p95), 2),
        "p99_ms": round(float(p99), 2),
        "p99_9_ms": round(float(p999), 2),
        "max_ms": round(float(max_lat), 2),
        "pass_p99_le_50ms": bool(p99 <= 50.0),
    }

    print("\nBenchmark Results:")
    print(f"Requests OK: {result['requests_ok']}")
    print(f"Errors:      {result['errors']}")
    print(f"Mean:        {result['mean_ms']:.2f} ms")
    print(f"P50:         {result['p50_ms']:.2f} ms")
    print(f"P95:         {result['p95_ms']:.2f} ms")
    print(f"P99:         {result['p99_ms']:.2f} ms")
    print(f"P99.9:       {result['p99_9_ms']:.2f} ms")
    print(f"Max:         {result['max_ms']:.2f} ms")
    
    if result["pass_p99_le_50ms"]:
        print(f"\nRESULT: PASS (P99={result['p99_ms']:.2f}ms <= 50ms)")
    else:
        print(f"\nRESULT: FAIL (P99={result['p99_ms']:.2f}ms > 50ms)")

    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--json-out")
    args = parser.parse_args()
    
    benchmark_result = run_benchmark(args.url, args.n, args.concurrency)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(benchmark_result, fh, indent=2)
        print(f"Saved JSON benchmark report to {args.json_out}")
