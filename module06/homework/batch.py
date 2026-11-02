"""NYC Yellow Taxi batch scoring with local filesystem or S3 storage.

Module 6 refactor: separate pure transformations, model inference, and I/O.
The CLI retains Module 4's local defaults; env templates override for integration tests.
"""

import argparse
import io
import os
import pickle
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HOMEWORK_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = PROJECT_ROOT / "module04" / "homework" / "model.bin"


def prepare_data(df: pd.DataFrame, categorical: list[str]) -> pd.DataFrame:
    """Convert categorical locations to strings and keep 1–60-minute trips."""
    result = df.copy()
    result["duration"] = (
        result["tpep_dropoff_datetime"] - result["tpep_pickup_datetime"]
    ).dt.total_seconds() / 60
    result = result[result["duration"].between(1, 60)].copy()
    for column in categorical:
        result[column] = result[column].fillna(-1).astype(int).astype(str)
    return result


def get_input_path(year: int, month: int) -> str:
    default_pattern = str(
        PROJECT_ROOT
        / "module04"
        / "homework"
        / "data"
        / "yellow_tripdata_{year:04d}-{month:02d}.parquet"
    )
    return os.getenv("INPUT_FILE_PATTERN", default_pattern).format(
        year=year, month=month
    )


def get_output_path(year: int, month: int) -> str:
    default_pattern = str(
        HOMEWORK_DIR
        / "output"
        / "yellow_tripdata_{year:04d}-{month:02d}_predictions.parquet"
    )
    return os.getenv("OUTPUT_FILE_PATTERN", default_pattern).format(
        year=year, month=month
    )


def parse_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.lstrip("/"):
        raise ValueError(f"Expected s3://bucket/key, got: {uri!r}")
    return parsed.netloc, parsed.path.lstrip("/")


def s3_client():
    """Use an explicit emulator endpoint when configured, else AWS SDK defaults."""
    import boto3
    from botocore.config import Config

    endpoint = os.getenv("S3_ENDPOINT_URL")
    return boto3.client(
        "s3",
        endpoint_url=endpoint or None,
        config=Config(s3={"addressing_style": "path"}) if endpoint else None,
    )


def read_data(path: str | Path, categorical: list[str]) -> pd.DataFrame:
    location = str(path)
    if location.startswith("s3://"):
        bucket, key = parse_s3_uri(location)
        response = s3_client().get_object(Bucket=bucket, Key=key)
        try:
            raw = response["Body"].read()
        finally:
            response["Body"].close()
        df = pd.read_parquet(io.BytesIO(raw), engine="pyarrow")
    else:
        df = pd.read_parquet(location, engine="pyarrow")
    return prepare_data(df, categorical)


def predict(df: pd.DataFrame, categorical: list[str], vectorizer, model):
    features = vectorizer.transform(df[categorical].to_dict(orient="records"))
    return model.predict(features)


def make_predictions(df: pd.DataFrame, predictions, year: int, month: int):
    """Preserve source DataFrame indices in ride IDs."""
    return pd.DataFrame(
        {
            "ride_id": f"{year:04d}/{month:02d}_" + df.index.astype(str),
            "predicted_duration": predictions,
        }
    )


def save_data(data: pd.DataFrame, path: str | Path) -> None:
    location = str(path)
    if location.startswith("s3://"):
        bucket, key = parse_s3_uri(location)
        buffer = io.BytesIO()
        data.to_parquet(buffer, engine="pyarrow", compression=None, index=False)
        s3_client().put_object(Bucket=bucket, Key=key, Body=buffer.getvalue())
    else:
        target = Path(location)
        target.parent.mkdir(parents=True, exist_ok=True)
        data.to_parquet(target, engine="pyarrow", compression=None, index=False)


def main(year: int, month: int) -> None:
    if not 1 <= month <= 12:
        raise ValueError("month must be between 1 and 12")

    categorical = ["PULocationID", "DOLocationID"]
    model_path = Path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL)))
    # Only load a known, trusted model artifact. pickle executes Python code.
    with model_path.open("rb") as file:
        vectorizer, model = pickle.load(file)

    input_path = get_input_path(year, month)
    output_path = get_output_path(year, month)
    df = read_data(input_path, categorical)
    predictions = predict(df, categorical, vectorizer, model)
    result = make_predictions(df, predictions, year, month)
    save_data(result, output_path)

    print("Rows:", len(result))
    print(f"Mean prediction: {predictions.mean():.4f}")
    print(f"Std prediction: {predictions.std():.4f}")
    print("Input:", input_path)
    print("Output:", output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Yellow Taxi batch inference")
    parser.add_argument("year", type=int)
    parser.add_argument("month", type=int)
    args = parser.parse_args()
    main(args.year, args.month)
