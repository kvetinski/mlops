"""Run with: python -m pytest -q module03/tests/test_reliability.py"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from time import sleep
from types import SimpleNamespace
from uuid import uuid4

import pytest
import requests

from module03.reliability import (
    FINGERPRINT_TAG,
    fingerprint_from_manifest,
    output_dir_for,
    quality_decision,
    register_once,
    is_retryable_health_error,
)


def fake_manifest(output_dir="aaa", data_hash="a"):
    return {
        "model_params": {"max_depth": 19},
        "environment": {"python": "3.12", "mlflow": "3.17.2"},
        "files": {
            "module02/data/green_tripdata_2023-01.parquet": {"sha256": data_hash},
            "module02/data/green_tripdata_2023-02.parquet": {"sha256": "b"},
            "module02/data/green_tripdata_2023-03.parquet": {"sha256": "c"},
            "module02/preprocess_data.py": {"sha256": "d"},
            "module03/pipeline.py": {"sha256": "e"},
            "module03/reliability.py": {"sha256": "g"},
            f"module03/output/{output_dir}/train.pkl": {"sha256": "f"},
        },
    }


def test_different_runs_have_different_directories(tmp_path):
    a, b = str(uuid4()), str(uuid4())
    assert output_dir_for(tmp_path, a) != output_dir_for(tmp_path, b)
    assert output_dir_for(tmp_path, a).parent == tmp_path
    with pytest.raises(ValueError):
        output_dir_for(tmp_path, "../../malicious")


def test_fingerprint_ignores_run_specific_paths():
    assert fingerprint_from_manifest(fake_manifest("run-one")) == fingerprint_from_manifest(fake_manifest("run-two"))
    assert fingerprint_from_manifest(fake_manifest("run-one")) != fingerprint_from_manifest(fake_manifest("run-two", "changed"))


def test_quality_gate_rejects_invalid_values():
    assert quality_decision(5.3354)
    assert not quality_decision(6.2)
    for value in (float("nan"), float("inf"), -1.0):
        with pytest.raises(ValueError):
            quality_decision(value)


class FakeRegistry:
    def __init__(self):
        self.versions = []
        self.created = 0
        self.guard = Lock()

    def search_model_versions(self, filter_string):
        with self.guard:
            return list(self.versions)

    def register(self, *, model_uri, name, tags):
        sleep(0.01)  # Make races more likely in the test
        with self.guard:
            self.created += 1
            result = SimpleNamespace(version=str(self.created), tags=dict(tags))
            self.versions.append(result)
            return result


def test_repeat_and_concurrent_registration_create_only_one_version(tmp_path):
    registry = FakeRegistry()

    def one_call(_):
        return register_once(
            client=registry, register_fn=registry.register, model_name="test-model",
            model_uri="models:/m-example", fingerprint="same-inputs", run_id="run-1",
            flow_id=str(uuid4()), lock_dir=tmp_path / "locks",
        )

    with ThreadPoolExecutor(max_workers=6) as pool:
        answers = list(pool.map(one_call, range(12)))

    assert registry.created == 1
    assert sum(created for _, created in answers) == 1
    assert {version.version for version, _ in answers} == {"1"}
    assert registry.versions[0].tags[FINGERPRINT_TAG] == "same-inputs"


def test_failed_register_can_be_safely_rerun(tmp_path):
    registry = FakeRegistry()
    attempts = 0

    def interrupted(*, model_uri, name, tags):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            # Simulate a lost response AFTER successful remote creation
            registry.register(model_uri=model_uri, name=name, tags=tags)
            raise TimeoutError("Response was lost")
        return registry.register(model_uri=model_uri, name=name, tags=tags)

    kwargs = dict(client=registry, register_fn=interrupted,
                  model_name="taxi", model_uri="models:/m-test",
                  fingerprint="same", run_id="run-1", flow_id=str(uuid4()),
                  lock_dir=tmp_path / "locks")
    with pytest.raises(TimeoutError):
        register_once(**kwargs)
    version, created = register_once(**kwargs)
    assert version.version == "1"
    assert not created
    assert registry.created == 1


def test_retry_selection():
    assert is_retryable_health_error(requests.exceptions.Timeout())
    assert is_retryable_health_error(requests.exceptions.ConnectionError())
    for status in (503, 429):
        response = requests.Response()
        response.status_code = status
        assert is_retryable_health_error(requests.exceptions.HTTPError(response=response))
    response = requests.Response()
    response.status_code = 401
    assert not is_retryable_health_error(requests.exceptions.HTTPError(response=response))
    assert not is_retryable_health_error(FileNotFoundError())
