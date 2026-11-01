import os
from datetime import date
from pathlib import Path

import pandas as pd
import psycopg


ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "module05/data/current.parquet"


def calculate_daily_metrics():
    df = pd.read_parquet(DATA_PATH)

    df["day"] = pd.to_datetime(df["trip_date"]).dt.date

    # Include only March 2024 observations.
    df = df[
        (df["day"] >= date(2024, 3, 1))
        & (df["day"] < date(2024, 4, 1))
    ]

    daily = df.groupby("day", as_index=False).agg(
        trips=("fare_amount", "size"),
        median_fare=("fare_amount", "median"),
        median_distance=("trip_distance", "median"),
        negative_fares=("negative_fare", "sum"),
        long_distance_trips=("distance_over_100_miles", "sum"),
    )

    return daily


def store_metrics(daily):
    database_url = os.environ["MONITORING_DATABASE_URL"]

    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS daily_data_quality (
                    day DATE PRIMARY KEY,
                    trips INTEGER NOT NULL,
                    median_fare DOUBLE PRECISION,
                    median_distance DOUBLE PRECISION,
                    negative_fares INTEGER NOT NULL,
                    long_distance_trips INTEGER NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
            """)

            for row in daily.itertuples(index=False):
                cur.execute("""
                    INSERT INTO daily_data_quality (
                        day, trips, median_fare, median_distance,
                        negative_fares, long_distance_trips
                    )
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (day) DO UPDATE SET
                        trips = EXCLUDED.trips,
                        median_fare = EXCLUDED.median_fare,
                        median_distance = EXCLUDED.median_distance,
                        negative_fares = EXCLUDED.negative_fares,
                        long_distance_trips =
                            EXCLUDED.long_distance_trips,
                        updated_at = NOW()
                """, (
                    row.day,
                    int(row.trips),
                    float(row.median_fare),
                    float(row.median_distance),
                    int(row.negative_fares),
                    int(row.long_distance_trips),
                ))


def main():
    daily = calculate_daily_metrics()

    if daily.empty:
        raise RuntimeError("No March 2024 records found")

    store_metrics(daily)

    peak = daily.loc[daily["median_fare"].idxmax()]

    print("Days monitored:", len(daily))
    print("Trips monitored:", daily["trips"].sum())
    print("Maximum daily median fare:", peak["median_fare"])
    print("Date:", peak["day"])
    print("Daily metrics saved to PostgreSQL")


if __name__ == "__main__":
    main()
