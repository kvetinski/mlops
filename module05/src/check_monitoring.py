"""Read-only verification of March 2024 taxi monitoring tables.

Usage (repository root):
    export MONITORING_DATABASE_URL='postgresql://monitoring:local_only_password@127.0.0.1:5433/monitoring'
    module05/.venv/bin/python module05/src/check_monitoring.py

No SQL mutations, model training, or monitoring metric computation.
"""

from collections import defaultdict
from datetime import date, timedelta
import math
import os
from statistics import mean

PERIOD_START = date(2024, 3, 1)
PERIOD_END = date(2024, 4, 1)
EXPECTED_DAYS = {PERIOD_START + timedelta(days=i) for i in range(31)}
FEATURES = frozenset({"PULocationID", "DOLocationID", "trip_distance", "fare_amount"})
REFERENCE_PERIOD = "2023-01"
EXPECTED_TRIPS = 57447
EXPECTED_REFERENCE_ROWS = 68211
EXPECTED_THRESHOLD = 0.2


def verify_quality(rows):
    """Rows: (day, trips, median_fare, median_distance, negative_fares, long_distance_trips)."""
    issues = []
    days = {}
    for day, trips, fare, distance, negative, long_distance in rows:
        if day in days:
            issues.append(f"Duplicate quality day: {day}")
        days[day] = trips
        if trips <= 0 or negative < 0 or long_distance < 0 or negative > trips or long_distance > trips:
            issues.append(f"Invalid quality counts: {day}")
        if not math.isfinite(float(fare)) or not math.isfinite(float(distance)):
            issues.append(f"Nonfinite daily median: {day}")
    if set(days) != EXPECTED_DAYS:
        issues.append(f"Quality calendar mismatch: missing={sorted(EXPECTED_DAYS - set(days))}; unexpected={sorted(set(days) - EXPECTED_DAYS)}")
    if sum(days.values()) != EXPECTED_TRIPS:
        issues.append(f"Expected {EXPECTED_TRIPS} March trips, found {sum(days.values())}")
    return issues, days


def verify_drift(rows, quality_by_day):
    """Rows: (day, feature, reference_period, method, threshold, score, detected, current_rows, reference_rows)."""
    issues = []
    by_day = defaultdict(set)
    scores = defaultdict(list)
    flagged = defaultdict(int)
    seen = set()
    for day, feature, period, method, threshold, score, detected, current_rows, reference_rows in rows:
        key = (day, feature)
        if key in seen:
            issues.append(f"Duplicate drift metric: {key}")
        seen.add(key)
        by_day[day].add(feature)
        if period != REFERENCE_PERIOD or method != "psi":
            issues.append(f"Unexpected drift configuration: {day} {feature}: {period} {method}")
        if not math.isclose(float(threshold), EXPECTED_THRESHOLD, abs_tol=1e-9):
            issues.append(f"Unexpected drift threshold: {day} {feature}: {threshold}")
        if feature not in FEATURES or not math.isfinite(float(score)) or score < 0:
            issues.append(f"Invalid drift feature/score: {day} {feature}: {score}")
        if bool(detected) != (score >= threshold):
            issues.append(f"Flag/threshold mismatch: {day} {feature}: {score} {detected}")
        if quality_by_day.get(day) != current_rows or reference_rows != EXPECTED_REFERENCE_ROWS:
            issues.append(f"Source-count mismatch: {day} {feature}")
        scores[feature].append((day, score))
        flagged[feature] += int(detected)

    if set(by_day) != EXPECTED_DAYS:
        issues.append(f"Drift calendar mismatch: missing={sorted(EXPECTED_DAYS - set(by_day))}")
    for day in EXPECTED_DAYS:
        if by_day[day] != FEATURES:
            issues.append(f"Drift feature coverage: {day} -> {sorted(by_day[day])}")
    if len(rows) != len(EXPECTED_DAYS) * len(FEATURES):
        issues.append(f"Expected 124 metrics, got {len(rows)}")
    return issues, scores, flagged


def main():
    dsn = os.getenv("MONITORING_DATABASE_URL")
    if not dsn:
        raise SystemExit("Set MONITORING_DATABASE_URL to your local PostgreSQL DSN")
    import psycopg
    with psycopg.connect(dsn, connect_timeout=10) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT day, trips, median_fare, median_distance,
                       negative_fares, long_distance_trips
                FROM daily_data_quality
                WHERE day >= %s AND day < %s ORDER BY day
            """, (PERIOD_START, PERIOD_END))
            quality = cur.fetchall()
            cur.execute("""
                SELECT day, feature, reference_period, method, threshold,
                       score, drift_detected, current_rows, reference_rows
                FROM daily_feature_drift
                WHERE day >= %s AND day < %s AND reference_period = %s
                ORDER BY day, feature
            """, (PERIOD_START, PERIOD_END, REFERENCE_PERIOD))
            drift = cur.fetchall()
    issues, daily_counts = verify_quality(quality)
    drift_issues, scores, flagged = verify_drift(drift, daily_counts)
    issues.extend(drift_issues)
    print(f"Data-quality days: {len(quality)}, trips: {sum(daily_counts.values())}")
    print(f"Drift metrics: {len(drift)}")
    for feature in sorted(scores):
        vals = scores[feature]
        weekday = [score for day, score in vals if day.weekday() < 5]
        weekend = [score for day, score in vals if day.weekday() >= 5]
        print(f"{feature:16s} avg_psi={mean(score for _, score in vals):.4f} "
              f"weekday={mean(weekday):.4f} weekend={mean(weekend):.4f} "
              f"drift_days={flagged[feature]}")
    if issues:
        for issue in issues:
            print("FAIL:", issue)
        raise SystemExit(f"Verification failed with {len(issues)} issue(s)")
    print("PASS: Complete March 2024 monitoring data; consistent PSI metrics and record counts")


if __name__ == "__main__":
    main()
