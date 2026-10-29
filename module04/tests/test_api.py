"""Black-box integration tests for the running taxi-duration inference API.

Start the container first. These tests do not download model artifacts or
interact with MLflow; they exercise the real HTTP server.
"""
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener

import pytest


BASE_URL = os.environ.get("TAXI_API_URL", "http://127.0.0.1:8001").rstrip("/")
TIMEOUT_SECONDS = 10
VALID_TRIP = {
    "PULocationID": 161,
    "DOLocationID": 141,
    "trip_distance": 3.5,
}

# Disable environment-based proxies for localhost/container tests.
_OPENER = build_opener(ProxyHandler({}))


def request_json(path: str, payload: dict | None = None) -> tuple[int, dict]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(
        f"{BASE_URL}{path}",
        data=data,
        headers={"Content-Type": "application/json"} if data is not None else {},
        method="POST" if data is not None else "GET",
    )
    try:
        with _OPENER.open(request, timeout=TIMEOUT_SECONDS) as response:
            return response.status, json.load(response)
    except HTTPError as exc:
        with exc:
            return exc.code, json.load(exc)


def test_health_check():
    status, result = request_json("/health")
    assert status == 200
    assert result["status"] == "ok"


def test_known_prediction_regression():
    status, result = request_json("/predict", VALID_TRIP)
    assert status == 200
    predicted = result["duration_minutes"]
    assert isinstance(predicted, (float, int))
    assert math.isfinite(predicted)
    assert predicted == pytest.approx(18.61, abs=0.05)


def test_prediction_is_deterministic():
    results = [request_json("/predict", VALID_TRIP) for _ in range(3)]
    assert all(status == 200 for status, _ in results)
    values = [result["duration_minutes"] for _, result in results]
    assert values[0] == pytest.approx(values[1], abs=1e-9)
    assert values[0] == pytest.approx(values[2], abs=1e-9)


@pytest.mark.parametrize("updates", [
    {"trip_distance": 0},
    {"trip_distance": -1},
    {"PULocationID": 0},
    {"DOLocationID": -1},
    {"trip_distance": "not-a-number"},
])
def test_invalid_values_return_422(updates):
    status, body = request_json("/predict", {**VALID_TRIP, **updates})
    assert status == 422, body
    assert "detail" in body


def test_missing_field_returns_422():
    incomplete_trip = {key: value for key, value in VALID_TRIP.items() if key != "trip_distance"}
    status, body = request_json("/predict", incomplete_trip)
    assert status == 422, body


def test_concurrent_requests_return_consistent_predictions():
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(lambda _: request_json("/predict", VALID_TRIP), range(24)))
    assert all(status == 200 for status, _ in results), results
    values = [result["duration_minutes"] for _, result in results]
    assert all(math.isfinite(value) for value in values)
    assert max(values) - min(values) < 1e-9
