"""Pure checks: tests do not need a running PostgreSQL/Grafana server."""
from datetime import date, timedelta

from module05.src.check_monitoring import (
    EXPECTED_DAYS, EXPECTED_TRIPS, EXPECTED_REFERENCE_ROWS,
    FEATURES, REFERENCE_PERIOD, verify_quality, verify_drift,
)


def sample_data():
    dates = sorted(EXPECTED_DAYS)
    quality = [
        (day, EXPECTED_TRIPS // 31 + (EXPECTED_TRIPS % 31 if i == 0 else 0),
         13.5, 1.79, 0, 0)
        for i, day in enumerate(dates)
    ]
    counts = {row[0]: row[1] for row in quality}
    drift = [
        (day, feature, REFERENCE_PERIOD, "psi", 0.2,
         0.25 if day.weekday() >= 5 else 0.12,
         day.weekday() >= 5, counts[day], EXPECTED_REFERENCE_ROWS)
        for day in dates for feature in sorted(FEATURES)
    ]
    return quality, drift


def test_complete_quality_and_drift_data_passes():
    quality, drift = sample_data()
    issues, counts = verify_quality(quality)
    assert issues == []
    issues, scores, flags = verify_drift(drift, counts)
    assert issues == []
    assert len(scores) == 4
    assert all(n == 10 for n in flags.values())


def test_missing_daily_quality_date_is_reported():
    quality, _ = sample_data()
    issues, _ = verify_quality(quality[:-1])
    assert any("calendar mismatch" in issue for issue in issues)


def test_duplicate_daily_drift_metric_is_reported():
    quality, drift = sample_data()
    _, counts = verify_quality(quality)
    issues, _, _ = verify_drift(drift + [drift[0]], counts)
    assert any("Duplicate drift metric" in issue for issue in issues)


def test_inconsistent_psi_flag_is_reported():
    quality, drift = sample_data()
    _, counts = verify_quality(quality)
    bad = list(drift[0]); bad[6] = not bad[6]
    drift[0] = tuple(bad)
    issues, _, _ = verify_drift(drift, counts)
    assert any("Flag/threshold mismatch" in issue for issue in issues)


def test_wrong_daily_counts_are_reported():
    quality, drift = sample_data()
    _, counts = verify_quality(quality)
    bad = list(drift[0]); bad[7] += 1
    drift[0] = tuple(bad)
    issues, _, _ = verify_drift(drift, counts)
    assert any("Source-count mismatch" in issue for issue in issues)
