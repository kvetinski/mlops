
import pickle
import shutil
import tempfile
from pathlib import Path

import mlflow
import mlflow.artifacts
import mlflow.sklearn


ROOT = Path(__file__).resolve().parent.parent

MODEL_URI = "models:/nyc-green-taxi-rf/2"
TRACKING_URI = "http://127.0.0.1:5000"

ARTIFACTS_DIR = ROOT / "module04" / "artifacts"
MODEL_DIR = ARTIFACTS_DIR / "model"
DV_SOURCE = ROOT / "module02" / "output" / "dv.pkl"
DV_TARGET = ARTIFACTS_DIR / "dv.pkl"


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # Download the registered model once.
    with tempfile.TemporaryDirectory() as tmp:
        downloaded = mlflow.artifacts.download_artifacts(
            artifact_uri=MODEL_URI,
            dst_path=tmp,
        )

        shutil.copytree(
            downloaded,
            MODEL_DIR,
            dirs_exist_ok=True,
        )

    # Package the fitted feature encoder.
    shutil.copy2(DV_SOURCE, DV_TARGET)

    # Verify that both artifacts are compatible.
    model = mlflow.sklearn.load_model(str(MODEL_DIR))

    with DV_TARGET.open("rb") as f:
        dv = pickle.load(f)

    actual = len(dv.get_feature_names_out())
    expected = model.n_features_in_

    if actual != expected:
        raise RuntimeError(
            f"Feature mismatch: {actual} != {expected}"
        )

    print("Model URI:", MODEL_URI)
    print("Model directory:", MODEL_DIR)
    print("Vectorizer:", DV_TARGET)
    print("Feature dimensions:", actual)
    print("Packaging successful")


if __name__ == "__main__":
    main()
