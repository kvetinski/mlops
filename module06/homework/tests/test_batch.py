from datetime import datetime

import pandas as pd

from module06.homework.batch import make_predictions, prepare_data


def dt(hour: int, minute: int, second: int = 0) -> datetime:
    return datetime(2023, 1, 1, hour, minute, second)


def sample_data() -> pd.DataFrame:
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


def test_prepare_data_filters_and_encodes() -> None:
    raw = sample_data()
    actual = prepare_data(raw, ["PULocationID", "DOLocationID"])

    assert len(actual) == 2
    assert actual.index.tolist() == [0, 1]
    assert actual[["PULocationID", "DOLocationID", "duration"]].to_dict(
        orient="records"
    ) == [
        {"PULocationID": "-1", "DOLocationID": "-1", "duration": 9.0},
        {"PULocationID": "1", "DOLocationID": "1", "duration": 8.0},
    ]
    # Pure transformation: input DataFrame must not be modified.
    assert "duration" not in raw.columns


def test_make_predictions_preserves_source_indices() -> None:
    df = prepare_data(sample_data(), ["PULocationID", "DOLocationID"])
    result = make_predictions(df, [8.5, 7.25], 2023, 1)

    assert result.to_dict(orient="records") == [
        {"ride_id": "2023/01_0", "predicted_duration": 8.5},
        {"ride_id": "2023/01_1", "predicted_duration": 7.25},
    ]
