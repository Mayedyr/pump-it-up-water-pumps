"""Train a reproducible starter model and create a competition submission."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from pump_it_up.data import ROOT, load_dataset

RANDOM_STATE = 7101


def add_date_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create stable calendar/age features and remove raw identifiers."""
    result = frame.copy()
    recorded = pd.to_datetime(result.pop("date_recorded"))
    result["recorded_year"] = recorded.dt.year
    result["recorded_month"] = recorded.dt.month
    result["waterpoint_age"] = (recorded.dt.year - result["construction_year"]).where(
        result["construction_year"].gt(0)
    )
    return result.drop(columns=["id"])


def build_pipeline(features: pd.DataFrame) -> Pipeline:
    """Build a leakage-safe mixed-type starter pipeline."""
    categorical = features.select_dtypes(include=["object", "string", "category"]).columns
    numeric = features.columns.difference(categorical)

    preprocessing = ColumnTransformer(
        transformers=[
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        (
                            "encode",
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                                encoded_missing_value=-1,
                            ),
                        ),
                    ]
                ),
                categorical,
            ),
            ("numeric", SimpleImputer(strategy="median"), numeric),
        ]
    )
    model = HistGradientBoostingClassifier(
        learning_rate=0.08,
        max_iter=250,
        l2_regularization=1.0,
        random_state=RANDOM_STATE,
    )
    return Pipeline([("preprocess", preprocessing), ("model", model)])


def run(cv_folds: int, output: Path) -> None:
    dataset = load_dataset()
    train = add_date_features(dataset.train_values)
    test = add_date_features(dataset.test_values)
    pipeline = build_pipeline(train)

    if cv_folds >= 2:
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
        scores = cross_val_score(
            pipeline, train, dataset.train_labels, cv=cv, scoring="accuracy", n_jobs=-1
        )
        print(f"CV accuracy: {scores.mean():.4f} +/- {scores.std():.4f}")

    pipeline.fit(train, dataset.train_labels)
    predictions = pipeline.predict(test)

    output.parent.mkdir(parents=True, exist_ok=True)
    submission = dataset.submission_format.copy()
    submission["status_group"] = predictions
    submission.to_csv(output, index=False)

    artifact = ROOT / "artifacts" / "baseline.joblib"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, artifact)
    print(f"Wrote {output}")
    print(f"Wrote {artifact}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cv-folds", type=int, default=5, help="Use 0 or 1 to skip CV")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "submissions" / "baseline.csv"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.cv_folds, args.output)

