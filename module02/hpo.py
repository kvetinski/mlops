
from pathlib import Path
import pickle

import mlflow
import numpy as np

from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.pyll import scope

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error


DATA_DIR = Path(__file__).resolve().parent / "output"
TRACKING_URI = "http://127.0.0.1:5000"


def load_pickle(filename):
    with open(DATA_DIR / filename, "rb") as f:
        return pickle.load(f)


def main():
    X_train, y_train = load_pickle("train.pkl")
    X_val, y_val = load_pickle("val.pkl")

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment("random-forest-hyperopt")

    def objective(params):
        # One MLflow run per trial
        with mlflow.start_run():
            mlflow.log_params(params)

            model = RandomForestRegressor(**params)
            model.fit(X_train, y_train)

            predictions = model.predict(X_val)
            rmse = root_mean_squared_error(
                y_val, predictions
            )

            mlflow.log_metric("rmse", float(rmse))

            print(f"Validation RMSE: {rmse:.4f}")
            
            return {
                "loss": rmse,
                "status": STATUS_OK,
            }

    search_space = {
        "max_depth": scope.int(
            hp.quniform("max_depth", 1, 20, 1)
        ),
        "n_estimators": scope.int(
            hp.quniform("n_estimators", 10, 50, 1)
        ),
        "min_samples_split": scope.int(
            hp.quniform("min_samples_split", 2, 10, 1)
        ),
        "min_samples_leaf": scope.int(
            hp.quniform("min_samples_leaf", 1, 4, 1)
        ),
        "random_state": 42,
    }

    trials = Trials()

    best = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=15,
        trials=trials,
        rstate=np.random.default_rng(42),
    )

    print("Best hyperparameters:", best)
    print("Best validation RMSE:", trials.best_trial["result"]["loss"])


if __name__ == "__main__":
    main()
