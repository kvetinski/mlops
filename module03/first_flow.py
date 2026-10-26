
from pathlib import Path
from prefect import flow, task
from prefect.cache_policies import NO_CACHE


DATA_DIR = (
    Path(__file__).resolve().parent.parent
    / "module02"
    / "output"
)


@task(
    name="check-artifact", 
    log_prints=True,     
    cache_policy=NO_CACHE,
)
def check_artifact(filename: str) -> tuple[str, int]:
    path = DATA_DIR / filename

    if not path.is_file():
        raise FileNotFoundError(path)

    size = path.stat().st_size

    print(f"{filename}: {size:,} bytes")

    return filename, size


@task(name="summarize-artifacts", log_prints=True)
def summarize_artifacts(
    artifacts: list[tuple[str, int]]
) -> int:

    total_size = sum(size for _, size in artifacts)

    print(f"Verified {len(artifacts)} artifacts")
    print(f"Total size: {total_size:,} bytes")

    return total_size


@flow(name="taxi-data-readiness", log_prints=True)
def taxi_data_readiness():

    filenames = [
        "train.pkl",
        "val.pkl",
        "test.pkl",
        "dv.pkl",
    ]

    artifacts = [
        check_artifact(filename)
        for filename in filenames
    ]

    total = summarize_artifacts(artifacts)

    return total


if __name__ == "__main__":
    taxi_data_readiness()
