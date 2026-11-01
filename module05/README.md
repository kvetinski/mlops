# Module 5 — Model monitoring for NYC Green Taxi

Reproducible, local monitoring of NYC Green Taxi trip data using **Evidently**, **PostgreSQL**, **Grafana**, and Docker Compose. Built as part of the DataTalksClub MLOps Zoomcamp 2025 and extended with feature-level drift diagnostics, idempotent database writes, and automated verification.

> **Scope:** This module monitors *historical taxi records*, not live predictions from the Module 4 FastAPI model. In particular, `fare_amount` is a contextual data-quality field, **not** an input feature of that model. No live model-quality claims should be inferred from these drift scores.

## Architecture

```text
Jan 2023 Green Taxi parquet ─> prepare_data.py ─> reference.parquet ───┐
                                                                        │
Mar 2024 Green Taxi parquet ─> prepare_data.py ─> current.parquet ──────┤
                                                                        ├─> monitor.py
                                                                        │   Evidently HTML + JSON
                                                                        │
                                                                        ├─> collect_drift.py
                                                                        │   daily PSI by feature
                                                                        │
                                                                        └─> collect_metrics.py
                                                                            daily quality metrics
                                                                                   │
                                                                                   v
                                                                              PostgreSQL
                                                                                   │
                                                                                   v
                                                                           Grafana dashboard
```

Daily collection is a batch replay over March 2024, not a continuously scheduled or streaming system.

## Datasets and baseline

| Dataset | Period | Rows |
|---|---|---:|
| Reference | January 2023 | 68,211 |
| Current | March 2024 (raw file) | 57,457 |
| Daily monitoring | March 1–31, 2024 pickup dates | 57,447 |

Ten raw-file trips have pickup dates outside March (seven in February 2024, two on 2008-12-31, one on 2024-04-01). They are preserved in `current.parquet` but excluded from March daily aggregates.

Features compared by Evidently PSI: `PULocationID`, `DOLocationID` (categorical); `trip_distance`, `fare_amount` (numerical). PSI threshold **0.2** is a heuristic for investigation, not a proof of prediction-quality degradation. The project also tracks negative fares and trips over 100 miles as simple data-quality indicators.

## Repository layout

```text
module05/
├── README.md
├── compose.yaml
├── config/
│   ├── grafana_datasources.yaml
│   └── grafana_dashboards.yaml
├── dashboards/
│   └── green_taxi_monitoring.json
├── src/
│   ├── prepare_data.py
│   ├── monitor.py
│   ├── collect_metrics.py
│   ├── collect_drift.py
│   └── check_monitoring.py
├── tests/
│   └── test_monitoring_checks.py
├── run_checks.sh
├── requirements.lock      # generated from the tested environment
├── data/                 # not committed
├── reports/              # generated, not committed
└── .venv/                # not committed
```

## Setup and execution

**Prerequisites:** Python 3.12, Docker with Compose, and the existing Jan 2023 Green Taxi parquet from Module 2.

1. Place March 2024 Green Taxi parquet at `module05/data/green_tripdata_2024-03.parquet` (available from the NYC TLC public trip-data source). The January 2023 file is `module02/data/green_tripdata_2023-01.parquet`.
2. Use an isolated Python environment with **Evidently 0.7.23**, Pandas, PyArrow, NumPy, and `psycopg[binary]` 3. `requirements.lock` should record the versions tested locally. Do not install the older Evidently 0.6 API beside the current code.
3. Run from the repository root:

```bash
cd ~/work/mlops
module05/.venv/bin/python -m pip check
module05/.venv/bin/python module05/src/prepare_data.py
module05/.venv/bin/python module05/src/monitor.py
xdg-open module05/reports/data_drift.html

docker compose -f module05/compose.yaml up -d
export MONITORING_DATABASE_URL='postgresql://monitoring:local_only_password@127.0.0.1:5433/monitoring'
module05/.venv/bin/python module05/src/collect_metrics.py
module05/.venv/bin/python module05/src/collect_drift.py
```

The sample password is **for local development only**. Use secrets/environment management for actual deployments, not committed credentials. The collector uses `ON CONFLICT ... DO UPDATE` on its day/feature/reference key to avoid duplicate records on rerun.

### Grafana

- Local UI: `http://127.0.0.1:3001`; datasource: `Monitoring PostgreSQL`.
- Dashboard is file-provisioned via `config/grafana_dashboards.yaml` and `dashboards/green_taxi_monitoring.json`.
- Set the time range to **March 1 through April 1, 2024** to see the historical data.
- Panels: daily median fare, negative-fare percentage, daily median distance, feature PSI, and number of drifted features.
- After UI edits, export **classic dashboard JSON** back into the repository: the provisioning system does not write edits into source files.

## Verified results (March 2024)

### Quality metrics

- Raw rows: **57,457**, rows with March pickup dates: **57,447** across **31 days**.
- Current vs reference median fare: **$13.50 vs $13.50**.
- Maximum of daily median fares: **$14.20 on March 3, 2024**.
- Reference vs current trips over 100 miles: **37 vs 20** (raw dataset counts).
- Reference vs current negative fares: **176 vs 196** (raw dataset counts).

### Evidently drift (daily PSI; baseline is all January 2023)

| Feature | Mean PSI | Maximum PSI | Days PSI ≥ 0.2 |
|---|---:|---:|---:|
| `PULocationID` | 0.2162 | 0.4096 | 12 / 31 |
| `DOLocationID` | 0.1924 | 0.2911 | 10 / 31 |
| `trip_distance` | 0.0245 | 0.0380 | 0 / 31 |
| `fare_amount` | 0.0122 | 0.0314 | 0 / 31 |

Total persisted: **124 feature/day measurements** across **31 days**.

### Weekday/weekend analysis

| Feature | Weekday mean PSI (21 days) | Weekend mean PSI (10 days) | Weekday drift | Weekend drift |
|---|---:|---:|---:|---:|
| `PULocationID` | 0.1540 | 0.3466 | 2 / 21 | 10 / 10 |
| `DOLocationID` | 0.1621 | 0.2559 | 0 / 21 | 10 / 10 |

Every weekend day triggered drift for both location features against the mixed-day baseline, consistent with **weekly seasonality**. This does **not** establish causality, nor prove model accuracy declined. A stronger next experiment uses day-type-matched reference distributions, and ultimately comparable seasonal windows, before escalating drift alerts.

## Tests and verification

The tests validate day/feature completeness, source-count consistency, uniqueness, threshold/flag agreement, and failure on corrupted data. The database verification is read-only.

```bash
cd ~/work/mlops
module05/.venv/bin/python -m pip install pytest
export MONITORING_DATABASE_URL='postgresql://monitoring:local_only_password@127.0.0.1:5433/monitoring'
bash module05/run_checks.sh
```

Or run database validation separately:

```bash
module05/.venv/bin/python module05/src/check_monitoring.py
```

The read-only check expects March 2024, **31 days of data quality**, **57,447 trips**, and **124 PSI metrics**. If data or parameters are deliberately changed, update the expected values and record the new experiment version instead of silently changing historic results.

## Course homework

The 2025 Module 5 exercises were completed through dataset inspection (Q1), the median-fare metric in Evidently (Q2), maximum daily median fare (Q3), and the exported Grafana dashboard (Q4). See the official [course homework](https://github.com/DataTalksClub/mlops-zoomcamp/blob/main/cohorts/2025/05-monitoring/homework.md).

