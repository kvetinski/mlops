"""Unit tests for configurable paths; no S3 service needed."""

import pytest

from module06.homework.batch import get_input_path, get_output_path, parse_s3_uri


def test_s3_patterns_are_applied(monkeypatch):
    monkeypatch.setenv(
        "INPUT_FILE_PATTERN", "s3://nyc-duration/in/{year:04d}-{month:02d}.parquet"
    )
    monkeypatch.setenv(
        "OUTPUT_FILE_PATTERN", "s3://nyc-duration/out/{year:04d}-{month:02d}.parquet"
    )
    assert get_input_path(2023, 1) == "s3://nyc-duration/in/2023-01.parquet"
    assert get_output_path(2023, 11) == "s3://nyc-duration/out/2023-11.parquet"


def test_valid_s3_uri():
    assert parse_s3_uri("s3://nyc-duration/in/file.parquet") == (
        "nyc-duration",
        "in/file.parquet",
    )


@pytest.mark.parametrize("bad", ["s3://", "s3://bucket", "s3:///x", "https://bucket/x"])
def test_invalid_s3_uri(bad):
    with pytest.raises(ValueError):
        parse_s3_uri(bad)
