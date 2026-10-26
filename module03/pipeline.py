"""Module 3: local Prefect + MLflow pipeline with reliability safeguards.

Run from ~/work/mlops:
    python module03/pipeline.py
    python module03/pipeline.py --register

Needs the existing 2023 Green Taxi Parquet files, Prefect API :4200,
and MLflow tracking server :5000. Pickle input must be trusted.
"""

import argparse
import pickle
import subprocess
import sys
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path

import mlflow
import mlflow.sklearn
import requests
from mlflow import MlflowClient
from prefect import flow, task
from prefect.cache_policies import NO_CACHE
from prefect.runtime import flow_run
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error

from reliability import (
    fingerprint_from_manifest,
    output_dir_for,
    quality_decision,
    register_once,
    is_retryable_health_error,
)

PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_DIR / "module02" / "data"
OUTPUT_ROOT = PROJECT_DIR / "module03" / "output"
LOCK_DIR = PROJECT_DIR / "module03" / ".locks"
PREPROCESS_SCRIPT = PROJECT_DIR / "module02" / "preprocess_data.py"

TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT = "taxi-training-pipeline"
MODEL_NAME = "nyc-green-taxi-rf"
THRESHOLD = 5.5  # Learning exercise, NOT a production model-promotion standard
MODEL_PARAMS = {
    "max_depth": 19,
    "n_estimators": 11,
    "min_samples_split": 2,
    "min_samples_leaf": 2,
    "random_state": 42,
}


def hash_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_DIR, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def should_retry_health_check(_task, _task_run, state) -> bool:
    """Retry only connection/timeouts or selected transient HTTP codes."""
    try:
        state.result()
    except Exception as exc:
        return is_retryable_health_error(exc)
    return False


@task(name="preprocess-data", retries=0, cache_policy=NO_CACHE, persist_result=False, log_prints=True)
def preprocess_data(output_dir: str) -> None:
    for month in ("01", "02", "03"):
        path = RAW_DATA_DIR / f"green_tripdata_2023-{month}.parquet"
        if not path.is_file():
            raise FileNotFoundError(f"Missing raw data: {path}")
    if not PREPROCESS_SCRIPT.is_file():
        raise FileNotFoundError(PREPROCESS_SCRIPT)

    Path(output_dir).mkdir(parents=True, exist_ok=False)
    subprocess.run(
        [sys.executable, str(PREPROCESS_SCRIPT),
         "--raw_data_path", str(RAW_DATA_DIR),
         "--dest_path", output_dir],
        cwd=PROJECT_DIR, check=True,
    )
    print(f"Preprocessing completed in {output_dir}")


@task(name="validate-datasets", retries=0, cache_policy=NO_CACHE)
def validate_datasets(output_dir: str) -> None:
    for filename in ("train.pkl", "val.pkl", "test.pkl", "dv.pkl"):
        path = Path(output_dir) / filename
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Missing or empty preprocessed artifact: {path}")


@task(name="capture-lineage", retries=0, cache_policy=NO_CACHE, persist_result=False)
def capture_lineage(output_dir: str) -> dict:
    paths = [RAW_DATA_DIR / f"green_tripdata_2023-{m}.parquet" for m in ("01", "02", "03")]
    paths += [Path(output_dir) / x for x in ("train.pkl", "val.pkl", "test.pkl", "dv.pkl")]
    paths += [PREPROCESS_SCRIPT, Path(__file__).resolve(), Path(__file__).with_name("reliability.py")]
    manifest = {
        "schema_version": 2,
        "git_revision": git_revision(),
        "model_params": MODEL_PARAMS,
        "environment": {
            "python": sys.version.split()[0],
            "scikit-learn": version("scikit-learn"),
            "mlflow": version("mlflow"),
            "prefect": version("prefect"),
        },
        "files": {},
    }
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        key = path.relative_to(PROJECT_DIR).as_posix()
        manifest["files"][key] = {"sha256": hash_file(path), "size_bytes": path.stat().st_size}
    manifest["pipeline_fingerprint"] = fingerprint_from_manifest(manifest)
    return manifest


@task(name="train-model", retries=0, cache_policy=NO_CACHE, persist_result=False, log_prints=True)
def train_model(output_dir: str) -> RandomForestRegressor:
    with (Path(output_dir) / "train.pkl").open("rb") as f:
        X_train, y_train = pickle.load(f)
    model = RandomForestRegressor(**MODEL_PARAMS)
    model.fit(X_train, y_train)
    print(f"Trained on {X_train.shape[0]:,} trips")
    return model


@task(name="evaluate-model", retries=0, cache_policy=NO_CACHE, log_prints=True)
def evaluate_model(model: RandomForestRegressor, output_dir: str) -> float:
    with (Path(output_dir) / "val.pkl").open("rb") as f:
        X_val, y_val = pickle.load(f)
    rmse = float(root_mean_squared_error(y_val, model.predict(X_val)))
    print(f"Validation RMSE: {rmse:.4f}")
    return rmse


@task(name="quality-gate", retries=0, cache_policy=NO_CACHE)
def quality_gate(rmse: float) -> bool:
    return quality_decision(rmse, THRESHOLD)


@task(name="check-mlflow-availability", retries=2, retry_delay_seconds=[2, 5],
      retry_condition_fn=should_retry_health_check, cache_policy=NO_CACHE, log_prints=True)
def check_mlflow_availability() -> None:
    # Read-only check; safe to retry. Registration itself is never auto-retried.
    response = requests.get(f"{TRACKING_URI}/health", timeout=5)
    response.raise_for_status()
    print("MLflow tracking server is reachable")


@task(name="log-and-optionally-register", retries=0, cache_policy=NO_CACHE,
      persist_result=False, log_prints=True)
def record_experiment(model: RandomForestRegressor, rmse: float, accepted: bool,
                      register: bool, lineage: dict, output_dir: str,
                      flow_id: str) -> dict:
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)

    with mlflow.start_run(run_name=f"prefect-{flow_id[:8]}") as run:
        mlflow.log_params(MODEL_PARAMS)
        mlflow.log_metric("val_rmse", rmse)
        mlflow.set_tags({
            "orchestrator": "prefect",
            "prefect_flow_run_id": flow_id,
            "pipeline_fingerprint": lineage["pipeline_fingerprint"],
            "quality_gate_passed": str(accepted).lower(),
            "registration_requested": str(register).lower(),
            "training_period": "2023-01",
            "validation_period": "2023-02",
        })
        mlflow.log_dict(lineage, "lineage/manifest.json")
        mlflow.log_artifact(str(Path(output_dir) / "dv.pkl"), artifact_path="preprocessing")
        model_info = mlflow.sklearn.log_model(
            sk_model=model, name="random-forest", serialization_format="skops",
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )
        run_id = run.info.run_id
        model_uri = model_info.model_uri
        result = {"run_id": run_id, "model_uri": model_uri, "registered_version": None,
                  "created_new_version": False}

        if accepted and register:
            client = MlflowClient(tracking_uri=TRACKING_URI)
            version_info, created = register_once(
                client=client,
                register_fn=mlflow.register_model,
                model_name=MODEL_NAME,
                model_uri=model_uri,
                fingerprint=lineage["pipeline_fingerprint"],
                run_id=run_id,
                flow_id=flow_id,
                lock_dir=LOCK_DIR,
            )
            result["registered_version"] = str(version_info.version)
            result["created_new_version"] = created

        mlflow.set_tags({
            "registered_version": result["registered_version"] or "none",
            "created_new_version": str(result["created_new_version"]).lower(),
            "logged_model_uri": model_uri,
        })
        print(result)
        return result


@flow(name="green-taxi-training-pipeline", log_prints=True, retries=0)
def training_pipeline(register: bool = False) -> dict:
    flow_id = str(flow_run.id)
    if not flow_id or flow_id == "None":
        raise RuntimeError("This pipeline requires a Prefect flow-run ID")
    work_dir = str(output_dir_for(OUTPUT_ROOT, flow_id))
    preprocess_data(work_dir)
    validate_datasets(work_dir)
    lineage = capture_lineage(work_dir)
    model = train_model(work_dir)
    rmse = evaluate_model(model, work_dir)
    accepted = quality_gate(rmse)
    check_mlflow_availability()
    result = record_experiment(model, rmse, accepted, register, lineage, work_dir, flow_id)
    return {"rmse": rmse, "accepted": accepted, "output_dir": work_dir, **result}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--register", action="store_true",
                        help="Register if quality gate passes; repeat data/code reuses existing version")
    args = parser.parse_args()
    training_pipeline(register=args.register)
