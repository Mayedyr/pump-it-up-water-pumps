"""Dataset loading and contract validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = ROOT / "data" / "raw"
TARGET = "status_group"


@dataclass(frozen=True)
class Dataset:
    train_values: pd.DataFrame
    train_labels: pd.Series
    test_values: pd.DataFrame
    submission_format: pd.DataFrame


def load_dataset(data_dir: Path = DEFAULT_DATA_DIR) -> Dataset:
    """Load the four competition files and fail fast on alignment problems."""
    train_values = pd.read_csv(data_dir / "train_values.csv", parse_dates=["date_recorded"])
    train_label_frame = pd.read_csv(data_dir / "train_labels.csv")
    test_values = pd.read_csv(data_dir / "test_values.csv", parse_dates=["date_recorded"])
    submission = pd.read_csv(data_dir / "submission_format.csv")

    if train_values["id"].duplicated().any() or test_values["id"].duplicated().any():
        raise ValueError("Feature IDs must be unique")
    if not train_values["id"].equals(train_label_frame["id"]):
        raise ValueError("Training feature and label IDs/order do not match")
    if not test_values["id"].equals(submission["id"]):
        raise ValueError("Test and submission IDs/order do not match")
    if list(train_values.columns) != list(test_values.columns):
        raise ValueError("Training and test feature schemas do not match")
    if set(train_label_frame[TARGET].unique()) != {
        "functional",
        "functional needs repair",
        "non functional",
    }:
        raise ValueError("Unexpected target classes")

    return Dataset(
        train_values=train_values,
        train_labels=train_label_frame[TARGET].rename(TARGET),
        test_values=test_values,
        submission_format=submission,
    )


def main() -> None:
    dataset = load_dataset()
    print(f"train: {dataset.train_values.shape}")
    print(f"test:  {dataset.test_values.shape}")
    print(dataset.train_labels.value_counts().to_string())
    print("Dataset contracts passed.")


if __name__ == "__main__":
    main()
