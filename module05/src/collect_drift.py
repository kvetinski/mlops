"""Calculate daily Evidently PSI drift and upsert results into PostgreSQL.

Run from repository root:
    module05/.venv/bin/python module05/src/collect_drift.py --day 2024-03-03 --dry-run
    module05/.venv/bin/python module05/src/collect_drift.py

Requires MONITORING_DATABASE_URL for database writes, not for --dry-run.
"""

import argparse
import math
import os
from datetime import date
from pathlib import Path

import pandas as pd
import psycopg
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "module05" / "data"
REFERENCE_PERIOD = "2023-01"
CURRENT_MONTH = "2024-03"
METHOD = "psi"
THRESHOLD = 0.2
CATEGORICAL = ["PULocationID", "DOLocationID"]
NUMERICAL = ["trip_distance", "fare_amount"]
FEATURES = CATEGORICAL + NUMERICAL


def normalize_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame[FEATURES].copy()
    for column in CATEGORICAL:
        result[column] = result[column].fillna(-1).astype(int).astype(str)
    return result


def extract_psi_metrics(snapshot: dict) -> dict[str, float]:
    """Read each column's ValueDrift result from snapshot.dict()."""
    values = {}
    for item in snapshot.get("metrics", []):
        if not item.get("metric_name", "").startswith("ValueDrift("):
            continue
        column = item.get("config", {}).get("column")
        if column not in FEATURES:
            continue
        if column in values:
            raise ValueError(f"Duplicate ValueDrift metric for {column}")
        try:
            value = float(item["value"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid ValueDrift metric: {item}") from exc
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"Non-finite or negative PSI for {column}: {value}")
        values[column] = value

    if set(values) != set(FEATURES):
        present = [m.get("metric_name") for m in snapshot.get("metrics", [])]
        raise ValueError(
            f"Expected PSI metrics for {FEATURES}, found {list(values)}. "
            f"All metric names: {present}"
        )
    return values


def collect(reference: pd.DataFrame, current: pd.DataFrame, requested_day: date | None):
    daily_dates = sorted(current["trip_day"].dropna().unique())
    if requested_day:
        daily_dates = [d for d in daily_dates if d == requested_day]
        if not daily_dates:
            raise ValueError(f"No records for requested date {requested_day}")

    definition = DataDefinition(
        categorical_columns=CATEGORICAL,
        numerical_columns=NUMERICAL,
    )
    ref_dataset = Dataset.from_pandas(reference, data_definition=definition)
    for day in daily_dates:
        current_day = normalize_features(current.loc[current["trip_day"] == day])
        if current_day.empty:
            continue
        day_dataset = Dataset.from_pandas(current_day, data_definition=definition)
        snapshot = Report([DataDriftPreset(method=METHOD, threshold=THRESHOLD)]).run(
            current_data=day_dataset,
            reference_data=ref_dataset,
        )
        metrics = extract_psi_metrics(snapshot.dict())
        yield (
            day,
            len(current_day),
            [(name, metrics[name], metrics[name] >= THRESHOLD) for name in FEATURES],
        )


def upsert_metrics(conn: psycopg.Connection, day: date, rows: int, reference_rows: int, metrics):
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS daily_feature_drift (
                day DATE NOT NULL,
                feature TEXT NOT NULL,
                reference_period TEXT NOT NULL,
                method TEXT NOT NULL,
                threshold DOUBLE PRECISION NOT NULL,
                score DOUBLE PRECISION NOT NULL,
                drift_detected BOOLEAN NOT NULL,
                current_rows INTEGER NOT NULL,
                reference_rows INTEGER NOT NULL,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                PRIMARY KEY (day, feature, reference_period)
            )
        """)
        cur.executemany("""
            INSERT INTO daily_feature_drift (
                day, feature, reference_period, method, threshold,
                score, drift_detected, current_rows, reference_rows
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (day, feature, reference_period) DO UPDATE SET
                method = EXCLUDED.method,
                threshold = EXCLUDED.threshold,
                score = EXCLUDED.score,
                drift_detected = EXCLUDED.drift_detected,
                current_rows = EXCLUDED.current_rows,
                reference_rows = EXCLUDED.reference_rows,
                updated_at = NOW()
        """, [
            (day, feature, REFERENCE_PERIOD, METHOD, THRESHOLD,
             score, detected, rows, reference_rows)
            for feature, score, detected in metrics
        ])


def main():
    parser = argparse.ArgumentParser(description="Daily taxi data drift monitoring")
    parser.add_argument("--day", type=date.fromisoformat, default=None,
                        help="Process only this YYYY-MM-DD date")
    parser.add_argument("--dry-run", action="store_true",
                        help="Calculate metrics without writing to PostgreSQL")
    args = parser.parse_args()
    if args.day and args.day.strftime("%Y-%m") != CURRENT_MONTH:
        parser.error(f"--day must fall in {CURRENT_MONTH}")

    reference = normalize_features(pd.read_parquet(DATA_DIR / "reference.parquet"))
    current = pd.read_parquet(DATA_DIR / "current.parquet")
    current["trip_day"] = pd.to_datetime(current["trip_date"]).dt.date
    current = current.loc[
        current["trip_day"].map(lambda d: d is not None and pd.notna(d)
                                and d.strftime("%Y-%m") == CURRENT_MONTH)
    ].copy()

    database_url = None if args.dry_run else os.environ.get("MONITORING_DATABASE_URL")
    if not args.dry_run and not database_url:
        parser.error("Set MONITORING_DATABASE_URL or use --dry-run")

    count_days = 0
    count_rows = 0
    if args.dry_run:
        conn = None
    else:
        conn = psycopg.connect(database_url)
    try:
        for day, rows, metrics in collect(reference, current, args.day):
            detected = sum(int(flag) for _, _, flag in metrics)
            print(f"{day}: rows={rows}, drifted={detected}/{len(FEATURES)}")
            for feature, score, flag in metrics:
                print(f"  {feature:<16s} PSI={score:8.4f}  drift={flag}")
            if conn is not None:
                upsert_metrics(conn, day, rows, len(reference), metrics)
                conn.commit()
            count_days += 1
            count_rows += len(metrics)
    finally:
        if conn is not None:
            conn.close()

    if not count_days:
        raise RuntimeError("No days processed")
    print(f"Completed: {count_days} day(s), {count_rows} metrics" +
          (" (dry run)" if args.dry_run else " stored in PostgreSQL"))


if __name__ == "__main__":
    main()
