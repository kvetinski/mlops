from pathlib import Path 
import pickle

import mlflow
import mlflow.sklearn

from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error

DATA_DIR = Path(__file__).resolve().parent / "output"

TRACKING_URI = "http://127.0.0.1:5000"
MODEL_NAME = "nyc-green-taxi-rf"

RF_PARAMS = [
    "max_depth",
    "n_estimators",
    "min_samples_split",
    "min_samples_leaf",
    "random_state",
]

def load_pickle(filename):
    with open(DATA_DIR / filename, "rb") as f:
        return pickle.load(f)

def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment("random-forest-best-models")

    client = MlflowClient()

    
    experiment = client.get_experiment_by_name(
        "random-forest-hyperopt"
    )

    if experiment is None:
        raise RuntimeError("HPO experiment not found")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="metrics.rmse >= 0",
        order_by=["metrics.rmse ASC"],
        max_results=1,
    )

    if not runs:
        raise RuntimeError("No completed HPO runs found")

    best_run = runs[0]
    best_run_id = best_run.info.run_id

    params = {
        key: int(best_run.data.params[key])
        for key in RF_PARAMS
    }

    # Load processed datasets
    X_train, y_train = load_pickle("train.pkl")
    X_val, y_val = load_pickle("val.pkl")
    X_test, y_test = load_pickle("test.pkl")

    with mlflow.start_run(run_name="selected-model") as run:
        mlflow.log_params(params)

        mlflow.set_tags({
            "source_hpo_run_id": best_run_id,
            "training_period": "2023-01",
            "validation_period": "2023-02",
            "test_period": "2023-03",
        })

        # Reproduce the selected model
        model = RandomForestRegressor(**params)
        model.fit(X_train, y_train)

        # Evaluate
        val_rmse = root_mean_squared_error(
            y_val, model.predict(X_val)
        )

        test_rmse = root_mean_squared_error(
            y_test, model.predict(X_test)
        )

        mlflow.log_metrics({
            "val_rmse": float(val_rmse),
            "test_rmse": float(test_rmse),
        })

        # Store the fitted feature encoder
        mlflow.log_artifact(
            str(DATA_DIR / "dv.pkl"),
            artifact_path="preprocessing",
        )

        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            name="random-forest",
            registered_model_name=MODEL_NAME,
            serialization_format="skops",
            skops_trusted_types=[
                "sklearn.tree._tree.Tree"
            ],
        )


        print(f"Validation RMSE: {val_rmse:.4f}")
        print(f"Test RMSE: {test_rmse:.4f}")
        print(f"Run ID: {run.info.run_id}")
        print(f"Model URI: {model_info.model_uri}")


if __name__ == "__main__":
    main()
