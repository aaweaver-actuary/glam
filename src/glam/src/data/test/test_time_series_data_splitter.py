"""Tests for the TimeSeriesDataSplitter class."""

import pytest
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal
import numpy as np
from glam.src.data.default_model_data import DefaultModelData
from glam.src.data.data_prep.splitters.time_series_data_splitter import (
    TimeSeriesDataSplitter,
)


@pytest.fixture
def numpy_rng() -> np.random.Generator:
    """Fixture to create a numpy random number generator."""
    return np.random.default_rng(42)


@pytest.fixture
def sample_time_series_data(numpy_rng: np.random.Generator) -> pd.DataFrame:
    """Fixture to create a sample time series data for testing."""
    return pd.DataFrame(
        {
            "feature1": list(range(100)),
            "feature2": [i % 5 for i in range(100)],
            "target": numpy_rng.integers(0, 2, 100),
            "cv": [i % 5 for i in range(100)],
        }
    )


@pytest.fixture
def base_model_data(sample_time_series_data: pd.DataFrame) -> DefaultModelData:
    """Fixture to create a DefaultModelData instance."""
    X = sample_time_series_data[["feature1", "feature2", "target", "cv"]]
    return DefaultModelData(df=X, y="target", cv="cv")


@pytest.fixture
def time_series_splitter(base_model_data: DefaultModelData) -> TimeSeriesDataSplitter:
    """Fixture to create a TimeSeriesDataSplitter instance."""
    return TimeSeriesDataSplitter(data=base_model_data)


def test_data_retrieval_properties(
    time_series_splitter: TimeSeriesDataSplitter, base_model_data: DefaultModelData
) -> None:
    """Test the data retrieval properties."""
    assert (
        time_series_splitter.data is base_model_data
    ), f"{time_series_splitter.data} != {base_model_data}"
    assert_frame_equal(time_series_splitter.X, base_model_data.X, check_dtype=False)
    assert_series_equal(time_series_splitter.y, base_model_data.y, check_dtype=False)
    assert_series_equal(time_series_splitter.cv, base_model_data.cv, check_dtype=False)

    assert time_series_splitter.fold_labels == sorted(
        base_model_data.cv.unique()
    ), f"{time_series_splitter.fold_labels} != {sorted(base_model_data.cv.unique())}"


@pytest.mark.parametrize(
    "fold,expected_train_size,expected_test_size",
    [
        (1, 20, 20),  # Fold 1: 1st fold as test, 0th fold as train
        (2, 40, 20),  # Fold 2: 1st and 2nd folds as train, 2nd fold as test
        (3, 60, 20),  # Fold 3: 1st, 2nd and 3rd folds as train, 3rd fold as test
        (4, 80, 20),  # Fold 4: 1st through 4th folds as train, 4th fold as test
    ],
)
def test_split_data(
    time_series_splitter: TimeSeriesDataSplitter,
    fold: int,
    expected_train_size: int,
    expected_test_size: int,
) -> None:
    """Test the split_data method."""
    X_train, y_train, X_test, y_test = time_series_splitter.split_data(fold)

    assert X_train.shape[0] == expected_train_size
    assert y_train.shape[0] == expected_train_size
    assert X_test.shape[0] == expected_test_size
    assert y_test.shape[0] == expected_test_size

    # Ensure that training data comes from earlier folds
    assert all(time_series_splitter.cv[X_train.index.tolist()] < fold)
    # Ensure that test data comes from the current fold
    assert all(time_series_splitter.cv[X_test.index.tolist()] == fold)


def test_fold_labels_order(time_series_splitter: TimeSeriesDataSplitter) -> None:
    """Test that fold_labels are sorted."""
    assert time_series_splitter.fold_labels == sorted(time_series_splitter.fold_labels)


def test_X_y_generator(time_series_splitter: TimeSeriesDataSplitter) -> None:
    """Test the X_y_generator method."""
    generator = time_series_splitter.X_y_generator
    fold_count = 1
    for X_train, y_train, X_test, y_test in generator:
        assert X_train.shape[0] == 20 * fold_count
        assert y_train.shape[0] == 20 * fold_count
        assert X_test.shape[0] == 20
        assert y_test.shape[0] == 20
        assert all(time_series_splitter.cv[X_train.index] < fold_count + 1)
        assert all(time_series_splitter.cv[X_test.index] == fold_count)
        fold_count += 1
