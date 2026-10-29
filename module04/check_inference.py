
from pathlib import Path
import pickle

import mlflow
import mlflow.sklearn


PROJECT_DIR = Path(__file__).resolve().parent.parent

MODEL_URI = "models:/nyc-green-taxi-rf/2"
TRACKING_URI = "http://127.0.0.1:5000"


def main():
    mlflow.set_tracking_uri(TRACKING_URI)

    # Load the trained Random Forest
    model = mlflow.sklearn.load_model(MODEL_URI)

    # Load the corresponding feature encoder
    vectorizer_path = (
        PROJECT_DIR / "module02" / "output" / "dv.pkl"
    )

    with vectorizer_path.open("rb") as f:
        dv = pickle.load(f)

    # Example raw taxi trip
    trip = {
        "PULocationID": 161,
        "DOLocationID": 141,
        "trip_distance": 3.5,
    }

    # Apply the same feature engineering as training
    features = {
        "PU_DO": (
            f"{trip['PULocationID']}_"
            f"{trip['DOLocationID']}"
        ),
        "trip_distance": trip["trip_distance"],
    }

    X = dv.transform([features])
    prediction = model.predict(X)

    print("Feature shape:", X.shape)
    print(f"Predicted duration: {prediction[0]:.2f} minutes")
    print("Model expects:", model.n_features_in_)
    print("Vectorizer produces:", X.shape[1])

if __name__ == "__main__":
    main()
