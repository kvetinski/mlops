"""End-to-end S3 batch-scoring test against local LocalStack only.

Run with the same Pipenv interpreter used for batch.py. Requires model.bin.
"""

import io
import os
import pickle
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import boto3
import numpy as np
import pandas as pd
from botocore.config import Config
from botocore.exceptions import ClientError

ROOT = Path(__file__).resolve().parents[2]
BATCH = Path(__file__).resolve().parent / "batch.py"
MODEL = ROOT / "module04" / "homework" / "model.bin"
ENDPOINT = os.getenv("S3_ENDPOINT_URL", "http://127.0.0.1:4566")
BUCKET = os.getenv("S3_BUCKET", "nyc-duration")
YEAR, MONTH = 2023, 1
INPUT_KEY = "in/2023-01.parquet"
OUTPUT_KEY = "out/2023-01.parquet"


def dt(hour: int, minute: int, second: int = 0) -> datetime:
    return datetime(2023, 1, 1, hour, minute, second)


def fixture() -> pd.DataFrame:
    return pd.DataFrame(
        [
            (None, None, dt(1, 1), dt(1, 10)),
            (1, 1, dt(1, 2), dt(1, 10)),
            (1, None, dt(1, 2, 0), dt(1, 2, 59)),
            (3, 4, dt(1, 2, 0), dt(2, 2, 1)),
        ],
        columns=[
            "PULocationID",
            "DOLocationID",
            "tpep_pickup_datetime",
            "tpep_dropoff_datetime",
        ],
    )


def main() -> None:
    # Fail closed: integration test must never connect to real AWS by accident.
    hostname = urlparse(ENDPOINT).hostname
    if hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise RuntimeError("Integration test requires a loopback S3 endpoint")

    s3 = boto3.client(
        "s3",
        endpoint_url=ENDPOINT,
        aws_access_key_id="test",
        aws_secret_access_key="test",
        region_name="us-east-1",
        config=Config(s3={"addressing_style": "path"}),
    )
    try:
        s3.head_bucket(Bucket=BUCKET)
    except ClientError as exc:
        if exc.response["ResponseMetadata"]["HTTPStatusCode"] == 404:
            s3.create_bucket(Bucket=BUCKET)
        else:
            raise

    source = fixture()
    buffer = io.BytesIO()
    source.to_parquet(buffer, engine="pyarrow", compression=None, index=False)
    raw = buffer.getvalue()
    s3.put_object(Bucket=BUCKET, Key=INPUT_KEY, Body=raw)
    assert s3.head_object(Bucket=BUCKET, Key=INPUT_KEY)["ContentLength"] == len(raw)
    print("Uploaded fixture bytes:", len(raw))

    environment = os.environ.copy()
    environment.update(
        {
            "S3_ENDPOINT_URL": ENDPOINT,
            "AWS_ACCESS_KEY_ID": "test",
            "AWS_SECRET_ACCESS_KEY": "test",
            "AWS_DEFAULT_REGION": "us-east-1",
            "AWS_EC2_METADATA_DISABLED": "true",
            "INPUT_FILE_PATTERN": f"s3://{BUCKET}/in/{{year:04d}}-{{month:02d}}.parquet",
            "OUTPUT_FILE_PATTERN": f"s3://{BUCKET}/out/{{year:04d}}-{{month:02d}}.parquet",
        }
    )

    subprocess.run(
        [sys.executable, str(BATCH), str(YEAR), str(MONTH)],
        cwd=str(ROOT),
        env=environment,
        check=True,
    )

    response = s3.get_object(Bucket=BUCKET, Key=OUTPUT_KEY)
    try:
        output = pd.read_parquet(io.BytesIO(response["Body"].read()))
    finally:
        response["Body"].close()

    assert output.columns.tolist() == ["ride_id", "predicted_duration"]
    assert output["ride_id"].tolist() == ["2023/01_0", "2023/01_1"]
    assert len(output) == 2
    assert np.isfinite(output["predicted_duration"].to_numpy()).all()

    # Independent oracle for exact predictions, based on fixture and course model.
    with MODEL.open("rb") as file:
        vectorizer, model = pickle.load(file)  # trusted course file
    features = [
        {"PULocationID": "-1", "DOLocationID": "-1"},
        {"PULocationID": "1", "DOLocationID": "1"},
    ]
    expected = model.predict(vectorizer.transform(features))
    np.testing.assert_allclose(
        output["predicted_duration"].to_numpy(), expected, rtol=1e-10
    )
    print("Prediction sum:", round(float(output["predicted_duration"].sum()), 4))
    print("S3 INTEGRATION TEST PASSED")


if __name__ == "__main__":
    main()
