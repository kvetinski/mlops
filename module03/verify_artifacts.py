"""Inspect ordinary run artifacts in MLflow 3.

Run: python module03/verify_artifacts.py
Or:  python module03/verify_artifacts.py --run-id <ID>
"""

import argparse
import json
from pathlib import Path

from mlflow import MlflowClient

TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "taxi-training-pipeline"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", help="Inspect a specific run (defaults to the newest run)")
    args = parser.parse_args()

    client = MlflowClient(tracking_uri=TRACKING_URI)
    run_id = args.run_id

    if run_id is None:
        experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
        if experiment is None:
            raise RuntimeError(f"Experiment not found: {EXPERIMENT_NAME}")
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["attributes.start_time DESC"],
            max_results=1,
        )
        if not runs:
            raise RuntimeError(f"No runs found in experiment: {EXPERIMENT_NAME}")
        run_id = runs[0].info.run_id

    run = client.get_run(run_id)
    print(f"Run ID:       {run_id}")
    print(f"Run status:   {run.info.status}")
    print(f"Artifact URI: {run.info.artifact_uri}")
    print(f"Model URI:    {run.data.tags.get('logged_model_uri', '<none recorded>')}")

    print("\nRun artifact root:")
    for item in client.list_artifacts(run_id):
        print(f"  {item.path}{'/' if item.is_dir else ''}")

    print("\nLineage directory:")
    lineage_items = client.list_artifacts(run_id, "lineage")
    for item in lineage_items:
        print(f"  {item.path}")

    if not any(item.path == "lineage/manifest.json" for item in lineage_items):
        print("\nNo manifest exists in THIS run. Verify that you reran the updated pipeline.")
        return

    # Download the small JSON file to verify its contents, not just its listing.
    downloaded = client.download_artifacts(run_id, "lineage/manifest.json")
    manifest = json.loads(Path(downloaded).read_text(encoding="utf-8"))
    print("\nDownloaded manifest:")
    print(f"  Schema version: {manifest['schema_version']}")
    print(f"  Git revision:   {manifest.get('git_revision')}")
    print(f"  Hashed files:   {len(manifest['files'])}")
    print(f"  Location:       {downloaded}")


if __name__ == "__main__":
    main()
