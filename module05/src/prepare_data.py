from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "module05" / "data"

DATASETS = {
    "reference": (
        ROOT / "module02" / "data"
        / "green_tripdata_2023-01.parquet"
    ),
    "current": (
        DATA_DIR / "green_tripdata_2024-03.parquet"
    ),
}

COLUMNS = [
    "lpep_pickup_datetime",
    "lpep_dropoff_datetime",
    "PULocationID",
    "DOLocationID",
    "trip_distance",
    "fare_amount",
]


def prepare(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path, columns=COLUMNS).copy()

    df["duration_minutes"] = (
        df["lpep_dropoff_datetime"]
        - df["lpep_pickup_datetime"]
    ).dt.total_seconds() / 60

    df["trip_date"] = (
        df["lpep_pickup_datetime"].dt.date
    )

    # Explicit data-quality indicators.
    df["distance_over_100_miles"] = (
        df["trip_distance"] > 100
    )

    df["negative_fare"] = (
        df["fare_amount"] < 0
    )

    return df


def main():
    for name, path in DATASETS.items():
        df = prepare(path)

        output_path = DATA_DIR / f"{name}.parquet"
        df.to_parquet(output_path, index=False)

        print(f"\n{name.upper()}")
        print("Rows:", len(df))
        print(
            "Median distance:",
            round(df["trip_distance"].median(), 3)
        )
        print(
            "Median fare:",
            round(df["fare_amount"].median(), 3)
        )
        print(
            "Trips over 100 miles:",
            int(df["distance_over_100_miles"].sum())
        )
        print(
            "Negative fares:",
            int(df["negative_fare"].sum())
        )
        print("Saved:", output_path)


if __name__ == "__main__":
    main()
