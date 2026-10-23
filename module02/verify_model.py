
from pathlib import Path
import pickle

import mlflow
import mlflow.sklearn

from sklearn.metrics import root_mean_squared_error


TRACKING_URI = "http://127.0.0.1:5000"
MODEL_URI = "models:/nyc-green-taxi-rf/2"

DATA_DIR = Path(__file__).resolve().parent / "output"


def main():
    mlflow.set_tracking_uri(TRACKING_URI)

    # Load the registered model, without training
    model = mlflow.sklearn.load_model(MODEL_URI)

    # Load our trusted, locally generated test dataset
    with open(DATA_DIR / "test.pkl", "rb") as f:
        X_test, y_test = pickle.load(f)

    # Generate predictions
    predictions = model.predict(X_test)

    # Verify the saved model reproduces our result
    rmse = root_mean_squared_error(y_test, predictions)

    print(f"Loaded model: {MODEL_URI}")
    print(f"Test RMSE: {rmse:.4f}")

    assert abs(rmse - 5.5674) < 0.01

    print("\nFirst 5 predictions:")
    for actual, predicted in zip(y_test[:5], predictions[:5]):
        print(
            f"Actual: {actual:.2f} | "
            f"Predicted: {predicted:.2f}"
        )

    print("\nModel verification passed!")


if __name__ == "__main__":
    main()
