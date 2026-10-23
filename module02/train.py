
from pathlib import Path
import pickle

import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error


DATA_DIR = Path(__file__).resolve().parent / "output"
TRACKING_URI = "http://127.0.0.1:5000"


def load_pickle(filename):
    with open(DATA_DIR / filename, "rb") as f:
        return pickle.load(f)


def main():
    # 1. Load preprocessed datasets
    X_train, y_train = load_pickle("train.pkl")
    X_val, y_val = load_pickle("val.pkl")

    print("Training samples:", X_train.shape)
    print("Validation samples:", X_val.shape)

    # 2. Configure MLflow
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment("zoomcamp-random-forest")

    # Automatically log sklearn parameters and model
    mlflow.sklearn.autolog()

    # 3. Start a tracked experiment run
    with mlflow.start_run(run_name="rf-baseline") as run:

        # 4. Train the model
        model = RandomForestRegressor(
            max_depth=10,
            random_state=0
        )

        model.fit(X_train, y_train)

        # 5. Evaluate on February data
        predictions = model.predict(X_val)

        rmse = root_mean_squared_error(
            y_val, predictions
        )

        # Validation RMSE is logged explicitly
        mlflow.log_metric("val_rmse", float(rmse))

        mlflow.set_tags({
            "train_data": "2023-01",
            "validation_data": "2023-02"
        })

        print(f"Validation RMSE: {rmse:.4f}")
        print(f"Run ID: {run.info.run_id}")


if __name__ == "__main__":
    main()
