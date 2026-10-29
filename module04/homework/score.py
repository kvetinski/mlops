from pathlib import Path
import pickle
import argparse
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent


CATEGORICAL = ["PULocationID", "DOLocationID"]


def read_data(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)

    df["duration"] = (
        df.tpep_dropoff_datetime
        - df.tpep_pickup_datetime
    ).dt.total_seconds() / 60

    df = df[
        (df.duration >= 1)
        & (df.duration <= 60)
    ].copy()

    df[CATEGORICAL] = (
        df[CATEGORICAL]
        .fillna(-1)
        .astype(int)
        .astype(str)
    )

    return df


def main(year: int, month: int):
    model_path = BASE_DIR / "model.bin"

    with model_path.open("rb") as f:
        dv, model = pickle.load(f)

    input_path = (
        BASE_DIR / "data"
        / f"yellow_tripdata_{year}-{month:02d}.parquet"
    )

    df = read_data(input_path)

    X = dv.transform(
        df[CATEGORICAL].to_dict(orient="records")
    )

    predictions = model.predict(X)

    df["ride_id"] = (
        f"{year:04d}/{month}_"
        + df.index.astype(str)
    )

    result = pd.DataFrame({
        "ride_id": df["ride_id"],
        "predicted_duration": predictions,
    })

    output_path = (
        BASE_DIR / "output"
        / f"yellow_tripdata_{year}-{month:02d}_predictions.parquet"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    result.to_parquet(
        output_path,
        engine="pyarrow",
        compression=None,
        index=False,
    )

    print(f"Rows: {len(result)}")
    print(f"Mean prediction: {predictions.mean():.4f}")
    print(f"Std prediction: {predictions.std():.4f}")
    print(f"Output: {output_path}")
    print(f"Output size: {output_path.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch taxi duration prediction"
    )
    parser.add_argument("year", type=int)
    parser.add_argument("month", type=int, choices=range(1, 13))

    args = parser.parse_args()
    main(args.year, args.month)
