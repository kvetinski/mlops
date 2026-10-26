"""Prefect retry/failure lab. No ML training or MLflow registration occurs.

Run from the project root after setting PREFECT_API_URL:
    python module03/failure_lab.py --case transient
    python module03/failure_lab.py --case permanent
    python module03/failure_lab.py --case rejected
"""

import argparse
import math

from prefect import flow, task
from prefect.cache_policies import NO_CACHE
from prefect.runtime import task_run


@task(
    name="simulate-transient-service",
    retries=2,
    retry_delay_seconds=1,
    log_prints=True,
    cache_policy=NO_CACHE,
    persist_result=False,
)
def transient_service() -> str:
    attempt = task_run.run_count
    print(f"External service attempt #{attempt}")

    if attempt == 1:
        raise ConnectionError("Simulated temporary network outage")

    print("External service recovered")
    return "connected"


@task(name="validate-required-file", retries=0, log_prints=True, cache_policy=NO_CACHE)
def validate_required_file() -> None:
    raise FileNotFoundError("Simulated missing training dataset")


@task(name="quality-gate-test", retries=0, log_prints=True, cache_policy=NO_CACHE)
def quality_gate_test(rmse: float) -> bool:
    if not math.isfinite(rmse) or rmse < 0:
        raise ValueError(f"Invalid RMSE: {rmse}")

    accepted = rmse <= 5.5
    print(f"RMSE {rmse:.2f}, accepted: {accepted}")
    return accepted


@task(name="simulate-registration", retries=0, log_prints=True, cache_policy=NO_CACHE)
def simulate_registration() -> None:
    # Intentionally no MLflow API call. This is a safe demonstration.
    print("SIMULATION ONLY: model would be registered")


@flow(name="retry-lab-transient", log_prints=True)
def transient_demo() -> None:
    transient_service()
    print("Recovered: flow completed")


@flow(name="retry-lab-permanent", log_prints=True)
def permanent_demo() -> None:
    validate_required_file()
    simulate_registration()  # Must never execute.


@flow(name="retry-lab-quality-rejection", log_prints=True)
def rejected_demo() -> None:
    accepted = quality_gate_test(6.2)
    if accepted:
        simulate_registration()
    else:
        print("Model rejected: registration skipped")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Explore Prefect retry and failure handling")
    parser.add_argument(
        "--case", required=True, choices=("transient", "permanent", "rejected")
    )
    args = parser.parse_args()

    {"transient": transient_demo, "permanent": permanent_demo, "rejected": rejected_demo}[
        args.case
    ]()
