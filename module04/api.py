
from contextlib import asynccontextmanager
from pathlib import Path
import pickle

import mlflow
import mlflow.sklearn

from fastapi import FastAPI
from pydantic import BaseModel, Field



ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
MODEL_DIR = ARTIFACTS_DIR / "model"
VECTORIZER_PATH = ARTIFACTS_DIR / "dv.pkl"

PROJECT_DIR = Path(__file__).resolve().parent.parent

TRACKING_URI = "http://127.0.0.1:5000"
MODEL_URI = "models:/nyc-green-taxi-rf/2"


class TaxiTrip(BaseModel):
    PULocationID: int = Field(gt=0)
    DOLocationID: int = Field(gt=0)
    trip_distance: float = Field(gt=0)


class PredictionResponse(BaseModel):
    duration_minutes: float



@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = mlflow.sklearn.load_model(
        str(MODEL_DIR)
    )

    with VECTORIZER_PATH.open("rb") as f:
        app.state.dv = pickle.load(f)

    expected = app.state.model.n_features_in_
    actual = len(app.state.dv.get_feature_names_out())

    if expected != actual:
        raise RuntimeError("Feature dimension mismatch")

    yield


app = FastAPI(
    title="NYC Taxi Duration Prediction",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_URI}


@app.post("/predict", response_model=PredictionResponse)
def predict(trip: TaxiTrip):
    features = {
        "PU_DO": (
            f"{trip.PULocationID}_{trip.DOLocationID}"
        ),
        "trip_distance": trip.trip_distance,
    }

    X = app.state.dv.transform([features])
    prediction = app.state.model.predict(X)[0]

    return PredictionResponse(
        duration_minutes=float(prediction)
    )
