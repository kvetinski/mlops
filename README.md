# MLOps Zoomcamp — Engineering Portfolio

A hands-on MLOps engineering project built while completing the [DataTalks.Club MLOps Zoomcamp](https://github.com/DataTalksClub/mlops-zoomcamp).

**Goal:** Learn to build, deploy, monitor, and maintain reproducible machine learning systems, applying production software engineering principles throughout.

## Progress

| Module | Topic                                | Status    |
| ------ | ------------------------------------ | --------- |
| 01     | Introduction to MLOps                | Completed |
| 02     | Experiment Tracking & Model Registry | Planned   |
| 03     | ML Pipeline Orchestration            | Planned   |
| 04     | Model Deployment                     | Planned   |
| 05     | Model Monitoring                     | Planned   |
| 06     | Engineering Best Practices           | Planned   |
| Final  | Independent MLOps Capstone           | Planned   |

## Module 01 — NYC Taxi Duration Prediction

**Objective:** Predict taxi trip duration from pickup and dropoff locations using a linear regression model.

### Dataset

- Source: [NYC Taxi & Limousine Commission](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)
- Training: January 2023 Yellow Taxi trips
- Validation: February 2023 Yellow Taxi trips
- Target: Trip duration in minutes

### Approach

1. Load Parquet data with Pandas.
2. Calculate trip duration from pickup and dropoff timestamps.
3. Filter trips to durations between 1 and 60 minutes.
4. Encode pickup and dropoff location IDs using `DictVectorizer`.
5. Train a sparse-feature linear regression model with scikit-learn.
6. Evaluate on a separate month's data without refitting the vectorizer.

### Results

| Metric             |         Result |
| ------------------ | -------------: |
| Training samples   |      3,009,173 |
| Validation samples |      2,855,951 |
| Encoded features   |            515 |
| Training RMSE      | 7.6493 minutes |
| Validation RMSE    | 7.8118 minutes |
| Training wall time |    ~61 seconds |

Validation RMSE is approximately 2.13% higher than training RMSE, indicating relatively stable performance across these two months.

The model uses only pickup and dropoff locations. It does not account for traffic, departure time, route characteristics, or location-pair interactions.

### Reproduce Locally

Requires Python 3.12.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r module01/requirements.txt

mkdir -p module01/data

curl -fL -o module01/data/yellow_tripdata_2023-01.parquet \
  https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-01.parquet

curl -fL -o module01/data/yellow_tripdata_2023-02.parquet \
  https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-02.parquet

cd module01/notebooks
jupyter lab
```

Open `01_exploration.ipynb` and execute all cells.

The raw datasets are not included in this repository.

## Planned Capstone — FraudWatch

An end-to-end ML risk-scoring platform for financial transactions, targeting:

- Reproducible model training and experiment tracking
- Model registry and deployment
- Containerized inference
- Monitoring, observability, and data-drift detection
- Automated testing and CI/CD
- Local-first infrastructure

**Planned stack:** Python, scikit-learn, MLflow, Docker, Prefect, Evidently, Prometheus, Grafana, with optional Go-based event ingestion.

## References

- [MLOps Zoomcamp](https://github.com/DataTalksClub/mlops-zoomcamp)
- [NYC TLC Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)

This is an independent learning and portfolio project.
