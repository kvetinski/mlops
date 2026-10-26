# Module 03 — ML Pipeline Orchestration

A **local-first, end-to-end training workflow** built for [DataTalks.Club MLOps Zoomcamp](https://github.com/DataTalksClub/mlops-zoomcamp). It orchestrates data preprocessing, Random Forest training, evaluation, experiment tracking, and *optional* model registration.

**Stack:** Python 3.12, Prefect 3, MLflow 3, scikit-learn, Pandas, NYC Green Taxi trip records. No cloud account is required.

## Architecture

```text
  NYC Green Taxi data, 2023-01/02/03 (local Parquet)
                         |
                         v
                  Prefect flow run
                         |
              +----------v------------+
              | preprocess and check  | -> private output/<flow-run-id>/
              +----------+------------+
                         |
                         v
                 lineage manifest
             (SHA-256 + environment)
                         |
                         v
                 train Random Forest
                         |
                         v
              evaluate on February data
                         |
                         v
              quality gate: RMSE <= 5.5
                         |
                         v
              MLflow run + model artifact
                         |
               --register requested?
                         |
             yes AND gate passed
                         |
                         v
       registration lock + fingerprint check
                         |
              create or reuse model version
```

A **normal run logs a new experiment and model artifact, but does not register a new version**. `--register` opts into conditional model registration. The 5.5-minute quality threshold is a teaching example, not a production-grade promotion policy.

## Dataset, model, and reported results

| Part       | Source                   | Role                                 |
| ---------- | ------------------------ | ------------------------------------ |
| Train      | Green Taxi January 2023  | Fit model                            |
| Validation | Green Taxi February 2023 | Evaluation / quality gate            |
| Test       | Green Taxi March 2023    | Reserved final holdout from Module 2 |

Feature engineering creates a pickup/dropoff category and includes trip distance. The selected Random Forest uses 11 trees, depth 19, `min_samples_leaf=2`, `min_samples_split=2`, and `random_state=42`.

Module 2 reference results: **5.3354 min validation RMSE** and **5.5674 min test RMSE**. The orchestrated pipeline is expected to produce a similar February validation result with the same inputs and dependency versions; do not represent it as a newly verified result until the flow has run successfully.

## Layout

```text
module03/
  pipeline.py               # Prefect flow: preprocessing -> model -> MLflow
  reliability.py            # fingerprint, quality gate, local registry lock
  verify_artifacts.py       # inspect run artifacts via MLflow API
  failure_lab.py            # transient/permanent/rejected simulations
  tests/
    test_reliability.py     # pure/offline tests
    test_integration.py     # read-only live service integration test
  run_checks.sh             # run preflight, tests, optional live verification
  requirements.lock         # generate from the tested local environment
  output/                   # generated per-run artifacts (gitignored)
  .locks/                   # local registry coordination (gitignored)
```

The preprocessing implementation comes from `module02/preprocess_data.py`; it is executed as a subprocess with the active Python interpreter.

## First-time local setup

From the repository root (`~/work/mlops`) with Python 3.12:

```bash
source .venv/bin/activate
# If provisioning a NEW environment, install the exact tested lock first:
# python -m pip install -r module03/requirements.lock

mkdir -p module02/data
for month in 01 02 03; do
  curl -fL --retry 3 \
    -o "module02/data/green_tripdata_2023-${month}.parquet" \
    "https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2023-${month}.parquet"
done
```

**Terminal A — MLflow:**

```bash
cd ~/work/mlops/module02
source ../.venv/bin/activate
mkdir -p artifacts
mlflow server --backend-store-uri sqlite:///mlflow.db \
  --artifacts-destination ./artifacts --host 127.0.0.1 --port 5000
```

**Terminal B — Prefect:**

```bash
cd ~/work/mlops
source .venv/bin/activate
prefect config set PREFECT_API_URL=http://127.0.0.1:4200/api
prefect server start
```

**Terminal C — pipeline / tests:**

```bash
cd ~/work/mlops
source .venv/bin/activate
python module03/pipeline.py                 # track a new run; DO NOT register
python module03/verify_artifacts.py        # inspect manifest in latest run
```

Dashboards: Prefect <http://127.0.0.1:4200>, MLflow <http://127.0.0.1:5000>.

To explicitly request registration:

```bash
python module03/pipeline.py --register
```

When inputs, code, configuration and recorded environment match an earlier fingerprint, the local registry guard attempts to reuse the existing version instead of creating another. Every invocation still preprocesses and retrains; it does **not** cache the fitted model.

## Validation and release checks

```bash
# Offline syntax, dependency and reliability checks
bash module03/run_checks.sh

# With BOTH local servers running, verify existing experiment artifacts
bash module03/run_checks.sh --live
```

A separate failure exercise is provided:

```bash
python module03/failure_lab.py --case transient  # retry -> completed
python module03/failure_lab.py --case permanent  # no retry -> failed (expected)
python module03/failure_lab.py --case rejected   # completed, registration skipped
```

The live integration test is read-only: it checks existing MLflow/Prefect records and never starts training or registers a new version. It expects at least one run from the **Step 3.7** pipeline with lineage schema 2.
