
from pathlib import Path

import pandas as pd

from evidently import Dataset, DataDefinition, Report
from evidently.metrics import QuantileValue
from evidently.presets import DataDriftPreset


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "module05" / "data"
REPORT_DIR = ROOT / "module05" / "reports"

CATEGORICAL = ["PULocationID", "DOLocationID"]
NUMERICAL = ["trip_distance", "fare_amount"]
COLUMNS = CATEGORICAL + NUMERICAL


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path, columns=COLUMNS).copy()

    for column in CATEGORICAL:
        df[column] = (
            df[column]
            .fillna(-1)
            .astype(int)
            .astype(str)
        )

    return df


def main():
    reference_df = load_data(DATA_DIR / "reference.parquet")
    current_df = load_data(DATA_DIR / "current.parquet")

    definition = DataDefinition(
        categorical_columns=CATEGORICAL,
        numerical_columns=NUMERICAL,
    )

    reference = Dataset.from_pandas(
        reference_df,
        data_definition=definition,
    )

    current = Dataset.from_pandas(
        current_df,
        data_definition=definition,
    )

    report = Report([
        QuantileValue(
            column="fare_amount",
            quantile=0.5,
        ),
        DataDriftPreset(),
    ])

    snapshot = report.run(
        current_data=current,
        reference_data=reference,
    )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    html_path = REPORT_DIR / "data_drift.html"
    json_path = REPORT_DIR / "data_drift.json"

    snapshot.save_html(str(html_path))
    snapshot.save_json(str(json_path))

    print("Reference rows:", len(reference_df))
    print("Current rows:", len(current_df))

    print(
        "Reference median fare:",
        reference_df["fare_amount"].median(),
    )
    print(
        "Current median fare:",
        current_df["fare_amount"].median(),
    )

    print("HTML report:", html_path)
    print("JSON report:", json_path)


if __name__ == "__main__":
    main()
