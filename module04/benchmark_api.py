"""Simple HTTP load generator for the local taxi prediction service.

This measures end-to-end request time (HTTP + validation + encoding + model
inference), not just Random Forest prediction time. Each request establishes
an HTTP connection; this is not a production-grade capacity benchmark.
"""

import argparse
import json
import math
import statistics
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener

TRIP = {"PULocationID": 161, "DOLocationID": 141, "trip_distance": 3.5}
_PAYLOAD = json.dumps(TRIP).encode("utf-8")
_OPENER = build_opener(ProxyHandler({}))


def predict(url: str, timeout: float) -> tuple[float, int | None, str | None]:
    request = Request(
        url,
        data=_PAYLOAD,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.perf_counter()
    status = None
    error = None
    try:
        with _OPENER.open(request, timeout=timeout) as response:
            status = response.status
            result = json.load(response)
            value = result["duration_minutes"]
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                error = f"invalid prediction: {value!r}"
    except HTTPError as exc:
        status = exc.code
        error = f"HTTP {exc.code}"
    except (URLError, TimeoutError, OSError, ValueError, KeyError, TypeError) as exc:
        error = f"{type(exc).__name__}: {exc}"
    return (time.perf_counter() - start) * 1000.0, status, error


def percentile(sorted_samples: list[float], quantile: float) -> float:
    """Nearest-rank percentile."""
    if not sorted_samples:
        raise ValueError("Cannot calculate percentile of empty sample")
    index = max(0, math.ceil(quantile * len(sorted_samples)) - 1)
    return sorted_samples[index]


def benchmark(url: str, requests: int, concurrency: int, warmup: int,
              timeout: float) -> dict:
    for index in range(warmup):
        _, status, error = predict(url, timeout)
        if status != 200 or error:
            raise RuntimeError(f"Warmup request {index + 1} failed: HTTP {status}, {error}")

    wall_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(predict, url, timeout) for _ in range(requests)]
        results = [future.result() for future in as_completed(futures)]
    elapsed = time.perf_counter() - wall_start

    successful = sorted(
        latency for latency, status, error in results if status == 200 and error is None
    )
    failures = Counter(
        error or f"HTTP {status}"
        for _, status, error in results
        if status != 200 or error is not None
    )
    data = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "url": url,
        "requests": requests,
        "concurrency": concurrency,
        "warmup_requests": warmup,
        "elapsed_seconds": round(elapsed, 3),
        "throughput_requests_per_second": round(requests / elapsed, 2),
        "successful_requests": len(successful),
        "failed_requests": requests - len(successful),
        "failure_reasons": dict(failures),
    }
    if successful:
        data["successful_latency_ms"] = {
            "mean": round(statistics.mean(successful), 2),
            "p50": round(percentile(successful, 0.50), 2),
            "p95": round(percentile(successful, 0.95), 2),
            "p99": round(percentile(successful, 0.99), 2),
            "max": round(successful[-1], 2),
        }
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8001")
    parser.add_argument("--requests", type=int, default=200)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    args = parser.parse_args()
    if args.requests < 1 or args.concurrency < 1 or args.warmup < 0 or args.timeout <= 0:
        parser.error("requests and concurrency must be >= 1; warmup >= 0; timeout > 0")

    report = benchmark(
        args.url.rstrip("/") + "/predict",
        args.requests,
        args.concurrency,
        args.warmup,
        args.timeout,
    )
    print(json.dumps(report, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Saved report: {args.output}")
    if report["failed_requests"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
