"""Read-only live integration test (no retraining or registration).

Enable deliberately: MLOPS_INTEGRATION=1 python -m pytest -q module03/tests/test_integration.py
Requires the local Prefect and MLflow servers and at least one Step 3.7 pipeline run.
"""

import json
import os
from pathlib import Path

import pytest


@pytest.mark.skipif(os.getenv("MLOPS_INTEGRATION") != "1", reason="Opt-in live test")
def test_latest_pipeline_run_has_artifacts_and_prefect_available():
    import requests
    from mlflow import MlflowClient

    assert requests.get("http://127.0.0.1:4200/api/health", timeout=5).ok
    assert requests.get("http://127.0.0.1:5000/health", timeout=5).ok

    client = MlflowClient(tracking_uri="http://127.0.0.1:5000")
    experiment = client.get_experiment_by_name("taxi-training-pipeline")
    assert experiment is not None
    runs = client.search_runs(
        [experiment.experiment_id], order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    assert runs
    latest = runs[0]
    assert latest.info.status == "FINISHED"
    assert "val_rmse" in latest.data.metrics
    lineage_items = client.list_artifacts(latest.info.run_id, "lineage")
    assert any(x.path == "lineage/manifest.json" for x in lineage_items)
    manifest_path = client.download_artifacts(latest.info.run_id, "lineage/manifest.json")
    manifest = json.loads(Path(manifest_path).read_text())
    assert manifest["schema_version"] == 2
    assert len(manifest["pipeline_fingerprint"]) == 64
    assert len(manifest["files"]) == 10
