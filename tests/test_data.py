from pump_it_up.data import load_dataset
from pump_it_up.train import add_date_features, build_pipeline


def test_dataset_contracts() -> None:
    dataset = load_dataset()
    assert dataset.train_values.shape == (59_400, 40)
    assert dataset.test_values.shape == (14_850, 40)
    assert len(dataset.train_labels) == len(dataset.train_values)


def test_pipeline_fits_small_sample() -> None:
    dataset = load_dataset()
    features = add_date_features(dataset.train_values.head(200))
    pipeline = build_pipeline(features)
    pipeline.fit(features, dataset.train_labels.head(200))
    predictions = pipeline.predict(features.head(10))
    assert len(predictions) == 10

