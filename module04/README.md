# Module 4 — Model Deployment: NYC Green Taxi Trip Duration

A locally deployable **FastAPI inference service** for predicting NYC green taxi trip duration. The service serves a Random Forest model trained and registered in Modules 2–3, loads a matching `DictVectorizer` on startup, and runs inside a Docker image **without requiring a live MLflow server**.

This module covers model packaging, online inference, input validation, Docker deployment, integration tests, and HTTP load benchmarking. It is a **custom online-serving extension** to the MLOps Zoomcamp work; the course's batch-inference exercise is a separate deliverable and is **not documented as completed here**.

## Status

| Component                                 | Status                                                 |
| ----------------------------------------- | ------------------------------------------------------ |
| Load registered model and encode raw trip | Verified locally                                       |
| Package model and fitted vectorizer       | Completed; offline inference verified                  |
| FastAPI `/health` and `/predict`          | Implemented; smoke test passed                         |
| Docker image (single Uvicorn worker)      | Built and smoke-tested on Ubuntu using host networking |
| Benchmark runs at concurrency 1, 4, 16    | Completed; plots shown below                           |
| API integration test suite                | Included; final pytest output not recorded here        |
| Official course batch deployment exercise | Not included yet                                       |

## Architecture

```text
                 BUILD / PACKAGING (requires MLflow)

    MLflow Model Registry                 Training artifacts
    nyc-green-taxi-rf / 2                 module02/output/dv.pkl
              |                                    |
              v                                    v
         module04/artifacts/model/     module04/artifacts/dv.pkl
                        \                 /
                         \               /
                          Docker image
                               |
                 RUNTIME (no MLflow server)
                               |
 Client --> POST /predict --> FastAPI --> feature engineering
                                      --> DictVectorizer.transform()
                                      --> RandomForest.predict()
                                      --> JSON duration_minutes

                   GET /health --> readiness response
```

At runtime, the model and vectorizer are loaded **once at application startup**, not once per request. The Docker image includes the MLflow Python package for local model deserialization, but does not need network access to the MLflow tracking server.

## Model and features

| Property                     | Value                                                                      |
| ---------------------------- | -------------------------------------------------------------------------- |
| Problem                      | Regression: trip duration in minutes                                       |
| Dataset                      | NYC Green Taxi, January–March 2023                                         |
| Training / validation / test | January / February / March 2023                                            |
| Estimator                    | `RandomForestRegressor`                                                    |
| MLflow registry reference    | `models:/nyc-green-taxi-rf/2`                                              |
| Feature encoder              | `DictVectorizer`                                                           |
| Input fields                 | `PULocationID`, `DOLocationID`, `trip_distance`                            |
| Encoded features             | **5,702** (5,701 observed route categories + 1 numerical distance feature) |
| Validation RMSE              | **5.3354 min** (previous model-selection result)                           |
| Test RMSE                    | **5.5674 min** (previous Module 2 evaluation)                              |

Feature engineering combines pickup and dropoff IDs into a categorical route identifier:

```python
features = {
    "PU_DO": f"{trip['PULocationID']}_{trip['DOLocationID']}",
    "trip_distance": trip["trip_distance"],
}
X = vectorizer.transform([features])
result = model.predict(X)[0]
```

**Compatibility requirement:** the model and vectorizer must come from the same training configuration. Matching 5,702 dimensions is a necessary check, but **not proof** that column meanings or feature order match. The most robust future improvement is to log the preprocessing transformer together with the estimator as one versioned inference pipeline.

## Repository layout

```text
module04/
├── api.py                         # FastAPI application
├── check_inference.py             # One-off prediction via MLflow registry
├── package_model.py               # Copy registered model + vectorizer locally
├── prepare_api_requirements.py    # Pin API-serving requirements
├── requirements-api.txt           # Generated, pinned serving dependencies
├── Dockerfile
├── .dockerignore
├── smoke_test.py                   # HTTP smoke check
├── benchmark_api.py               # HTTP load generator; JSON reports
├── plot_benchmarks.py             # Plot latency, throughput, failure rate
├── tests/
│   └── test_api.py                # HTTP integration tests
├── docs/benchmark_plots/         # Benchmark figures committed to Git
│   ├── latency_vs_concurrency.png
│   ├── throughput_vs_concurrency.png
│   └── error_rate_vs_concurrency.png
└── artifacts/                     # Local model assets; ignored by Git
    ├── model/
    └── dv.pkl
```

## API contract

### `GET /health`

Returns HTTP 200 when the application has completed startup and is serving requests:

```json
{"status": "ok", "model": "models:/nyc-green-taxi-rf/2"}
```

The `model` field is descriptive metadata; the deployed instance actually loads the **packaged local artifact**.

### `POST /predict`

Example:

```bash
curl --noproxy '*' -X POST http://127.0.0.1:8001/predict \
  -H 'Content-Type: application/json' \
  -d '{"PULocationID":161,"DOLocationID":141,"trip_distance":3.5}'
```

Approximate response for the tested artifacts:

```json
{"duration_minutes": 18.61}
```

Location IDs must be positive integers; `trip_distance` must be positive. FastAPI/Pydantic rejects invalid requests with **HTTP 422**. The output is an estimated duration, **not** a guarantee of actual travel time. No authentication or request rate limiting has been added.

## Reproduce locally

Commands assume the repository root is `~/work/mlops` and a Python virtual environment exists there.

### 1. Package the model (one-time step)

With the MLflow server available at `http://127.0.0.1:5000` and the original `module02/output/dv.pkl` present:

```bash
cd ~/work/mlops
source .venv/bin/activate
python module04/package_model.py
```

This downloads registered model version 2, copies the fitted vectorizer, and verifies feature dimensions. **Only load the pickle and serialized model from trusted sources.**

### 2. Start the API without MLflow

Stop the MLflow server to verify runtime independence:

```bash
uvicorn module04.api:app --host 127.0.0.1 --port 8001
```

Open `http://127.0.0.1:8001/docs` for interactive API documentation.

### 3. Docker deployment

```bash
python module04/prepare_api_requirements.py

# On machines where Docker bridge DNS fails, host networking can help builds.
docker build --network=host -t taxi-duration-api:local module04

# Verified development workaround for this Ubuntu/VPN network setup:
docker run --rm --name taxi-duration-host --network host \
  taxi-duration-api:local \
  python -m uvicorn module04.api:app \
    --host 127.0.0.1 --port 8001
```

In a second terminal:

```bash
python module04/smoke_test.py
```

**Networking note:** the initial Docker bridge deployment exposed `127.0.0.1:8001:8000`, but host-to-container requests timed out while the internal Docker health check returned HTTP 200. Using `--network host` fixed connectivity and the smoke test passed. This is a **development-specific workaround**, not evidence that the underlying Docker/VPN routing issue is resolved. Host networking also reduces network isolation; investigate bridge networking before production deployment.

On a working bridge network, the ordinary alternative is:

```bash
docker run --rm --name taxi-duration \
  -p 127.0.0.1:8001:8000 taxi-duration-api:local
```

The image runs a single Uvicorn worker by default and uses a non-root user.

## Testing

Smoke test (verified with host networking):

```bash
python module04/smoke_test.py
```

The test checks `GET /health`, a known prediction (~18.61 min), and rejection of a negative distance (HTTP 422).

HTTP integration tests (included; run against the live service):

```bash
TAXI_API_URL=http://127.0.0.1:8001 \
  python -m pytest -v module04/tests/test_api.py
```

Ten test cases cover health, regression output, deterministic results, invalid input, missing fields, and a batch of 24 concurrent requests. **Do not report these tests as passing until the actual local pytest run has succeeded.**

## HTTP performance benchmark

Three local runs used **200 measured requests each**, with 10 warmup requests by default, at concurrency levels **1, 4, and 16**. The same example trip was submitted repeatedly. The benchmark uses a Python thread pool and opens a new HTTP connection per request; results include network, HTTP handling, validation, vectorization, and model prediction.

Run and reproduce the reports:

```bash
python module04/benchmark_api.py --requests 200 --concurrency 1  --output module04/reports/c1.json
python module04/benchmark_api.py --requests 200 --concurrency 4  --output module04/reports/c4.json
python module04/benchmark_api.py --requests 200 --concurrency 16 --output module04/reports/c16.json
python module04/plot_benchmarks.py
```

### Observed results

The following values are **approximations read from the supplied plots**; use the corresponding `c*.json` files when reporting exact percentiles.

| Concurrency | Throughput (requests/s) | p50 (ms) | p95 (ms) | p99 (ms) | HTTP failures |
| ----------: | ----------------------: | -------: | -------: | -------: | ------------: |
|           1 |                    ~461 |     ~2.0 |     ~3.0 |     ~3.4 |            0% |
|           4 |                    ~469 |     ~8.5 |    ~10.4 |    ~11.7 |            0% |
|          16 |                    ~479 |      ~33 |      ~39 |      ~44 |            0% |

![HTTP latency percentiles vs concurrency](reports/latency_vs_concurrency.png)

![HTTP throughput vs concurrency](reports/throughput_vs_concurrency.png)

![HTTP failure rate vs concurrency](reports/error_rate_vs_concurrency.png)

### Interpretation

- **Throughput was nearly flat:** increasing concurrency from 1 to 16 improved measured throughput by only about **4%** (~461 → ~479 requests/s).
- **Latency increased sharply:** p50 rose from about **2 ms to 33 ms**, while p99 rose from about **3–4 ms to 44 ms**. The service handled more outstanding requests without a corresponding increase in processing capacity.
- **No failures were observed** in these short runs. That does not establish availability under long-duration load, resource exhaustion, or higher concurrency.
- The throughput chart's vertical axis starts near **460**, rather than zero, making the small relative gain look larger visually.

These observations are **consistent with an early throughput bottleneck or load-generator limit**; the plots alone do not prove CPU saturation or identify the bottleneck. Collect container CPU/memory metrics, repeat trials with more requests, compare connection reuse and a stronger load generator, and profile preprocessing versus prediction before drawing capacity conclusions.

### Benchmark limitations

- Single local machine and one Uvicorn worker, with no independent load-generator host.
- Only 200 requests per concurrency level; particularly noisy for p99 estimates.
- One repeated route and distance, so no distribution of real-world inputs.
- Percentiles are computed from **successful HTTP requests only**; failure rate is reported separately.
- The benchmark throughput is **all completed requests / wall-clock duration**, rather than successful-only throughput.
- No sustained soak test, memory-leak analysis, or cold-start measurement.

## Next improvements

1. Run the full pytest suite and store the summary.
2. Repeat the load test with multiple trials and larger sample sizes; capture CPU/memory and report median throughput with spread.
3. Package the fitted vectorizer as part of the registered model itself, with a shared version/fingerprint.
4. Add structured request metrics, logging, tracing, and readiness checks.
5. Complete the course's batch-inference exercise and document it separately.
6. Add a CI job for syntax checks, tests, Docker builds, and artifact provenance checks.

## Git hygiene

Keep raw data, local model assets, virtual environments, caches, and generated JSON reports out of Git. Commit **source, Dockerfile, pinned dependency manifests, README, and the three benchmark PNG files** so the performance evidence renders in GitHub.

Suggested exclusions:

```gitignore
module04/artifacts/
module04/reports/
module04/__pycache__/
module04/tests/__pycache__/
```

The documentation and plots are reproducible from the original JSON reports; those reports should be retained locally or in a suitable versioned artifact store if exact benchmark auditability is required.
