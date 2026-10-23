# Module 02 — Experiment Tracking and Model Management

## Objective

Build a reproducible machine learning experimentation workflow using MLflow, scikit-learn, and Hyperopt.

Part of the [DataTalks.Club MLOps Zoomcamp](https://github.com/DataTalksClub/mlops-zoomcamp).

## Architecture

The workflow consists of:

1. Data preprocessing — convert raw taxi records into training, validation, and test matrices.
2. Model training — fit a Random Forest regressor.
3. Experiment tracking — record parameters, metrics, and artifacts using MLflow.
4. Hyperparameter optimization — run 15 Hyperopt trials.
5. Model selection — rank candidates by validation RMSE.
6. Model registration — save and version the selected model.
7. Verification — reload the registered model and reproduce predictions.

## Dataset

NYC Green Taxi trip records, 2023.

| Dataset    | Period   |
| ---------- | -------- |
| Training   | January  |
| Validation | February |
| Testing    | March    |

Features include pickup–dropoff location combinations and trip distance.

Target: trip duration in minutes.

## Results

| Metric                    |             Value |
| ------------------------- | ----------------: |
| Baseline validation RMSE  |          ~5.4 min |
| Optimized validation RMSE |        5.3354 min |
| Test RMSE                 |        5.5674 min |
| Hyperopt trials           |                15 |
| Registered model          | nyc-green-taxi-rf |
| Verified model version    |                 2 |

### Selected hyperparameters

```python
{
    "max_depth": 19,
    "n_estimators": 11,
    "min_samples_split": 2,
    "min_samples_leaf": 2,
    "random_state": 42,
}
```

## Running locally

Use the Python environment described in `requirements.lock`.

Download the January–March 2023 Green Taxi Parquet datasets from the NYC TLC trip data repository into `module02/data`.

Preprocess the datasets:

```bash
python module02/preprocess_data.py \
  --raw_data_path module02/data \
  --dest_path module02/output
```

Start MLflow from the `module02` directory:

```bash
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --artifacts-destination ./artifacts \
  --host 127.0.0.1 \
  --port 5000
```

From the repository root, run:

```bash
python module02/train.py
python module02/hpo.py
python module02/select_model.py
python module02/register_model.py
python module02/verify_model.py
```

The verification script currently references registered model version 2 from the original experiment. On a fresh installation, update its model URI to the version created locally.

## Engineering considerations

- Training, validation, and test datasets are separated chronologically.
- MLflow records parameters, training metrics, validation metrics, and model lineage.
- The held-out test dataset is evaluated after selecting the best model.
- Serialized models and training datasets are excluded from Git.
- Loading serialized Python objects requires trusting their source.
- The current model expects preprocessed feature matrices; preprocessing and inference are not yet packaged into a single serving pipeline.

## Future improvements

- Automated retraining orchestration.
- Deployment-ready preprocessing and inference pipeline.
- Model promotion using MLflow aliases.
- CI/CD and automated model evaluation.
- Drift detection and monitoring.
