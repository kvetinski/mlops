"""Pure helpers for the local, Linux-only MLOps Zoomcamp pipeline.

Registration de-duplication is coordinated with a POSIX file lock.
This coordinates processes on ONE host sharing the same lock directory,
not distributed workers on independent machines.
"""

import fcntl
import hashlib
import json
import math
import re
from contextlib import contextmanager
from pathlib import Path

import requests


FINGERPRINT_TAG = "pipeline_fingerprint"


def is_retryable_health_error(exc: Exception) -> bool:
    """Limit automatic Prefect retries to transient HTTP/network errors."""
    if isinstance(exc, (requests.exceptions.ConnectionError, requests.exceptions.Timeout)):
        return True
    if isinstance(exc, requests.exceptions.HTTPError):
        return exc.response is not None and exc.response.status_code in {429, 500, 502, 503, 504}
    return False


def output_dir_for(output_root: Path, flow_id: str) -> Path:
    """Give every Prefect flow run a separate output directory."""
    if not re.fullmatch(r"[0-9a-fA-F-]{36}", flow_id):
        raise ValueError(f"Unexpected flow run ID: {flow_id!r}")
    return Path(output_root) / flow_id


def quality_decision(rmse: float, threshold: float = 5.5) -> bool:
    if not math.isfinite(rmse) or rmse < 0:
        raise ValueError(f"Invalid RMSE: {rmse}")
    return rmse <= threshold


def fingerprint_from_manifest(manifest: dict) -> str:
    """Fingerprint the inputs, source, config and environment, NOT run paths.

    Generated files are retained in the lineage manifest, but are intentionally
    omitted from the key so output path names do not change de-duplication.
    """
    stable_files = {
        path: value["sha256"]
        for path, value in manifest["files"].items()
        if path.startswith("module02/data/")
        or path in ("module02/preprocess_data.py", "module03/pipeline.py", "module03/reliability.py")
    }
    if len(stable_files) != 6:
        raise ValueError("Fingerprint requires 3 raw datasets and 3 code files")

    payload = {
        "model_params": manifest["model_params"],
        "environment": manifest["environment"],
        "files": stable_files,
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


@contextmanager
def local_registry_lock(lock_dir: Path, model_name: str):
    """A process-safe lock for one registry name on a shared local filesystem."""
    lock_dir = Path(lock_dir)
    lock_dir.mkdir(parents=True, exist_ok=True)
    safe_name = hashlib.sha256(model_name.encode()).hexdigest()[:24]
    with (lock_dir / f"{safe_name}.lock").open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def register_once(*, client, register_fn, model_name: str, model_uri: str,
                  fingerprint: str, run_id: str, flow_id: str,
                  lock_dir: Path):
    """Reuse a registered version with the same fingerprint if it exists.

    register_fn must accept keyword args: model_uri, name, tags.
    For MLflow 3 pass mlflow.register_model. The tags are supplied IN the
    create call, not written afterward, to aid recovery from lost responses.
    """
    with local_registry_lock(lock_dir, model_name):
        versions = client.search_model_versions(f"name='{model_name}'")
        for version in versions:
            if version.tags.get(FINGERPRINT_TAG) == fingerprint:
                return version, False

        version = register_fn(
            model_uri=model_uri,
            name=model_name,
            tags={
                FINGERPRINT_TAG: fingerprint,
                "source_mlflow_run_id": run_id,
                "source_prefect_flow_run_id": flow_id,
            },
        )
        return version, True
