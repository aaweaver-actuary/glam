import pytest
import pandas as pd
from glam.src.data.data_prep.preprocessors.default_preprocessor import (
    DefaultPreprocessor,
)
from glam.src.data.default_model_data import DefaultModelData


@pytest.fixture
def sample_data():
    data = {
        "numeric_col": [1, 2, None, 4, 5],
        "categorical_col": ["a", "b", None, "a", "b"],
        "fold": [0, 1, 2, 2, 1],
        "target": [0, 1, 0, 1, 0],
    }
    df = pd.DataFrame(data)
    return DefaultModelData(df)


def test_default_preprocessor_init(sample_data):
    preprocessor = DefaultPreprocessor(sample_data)

    # Check if the data attribute is set correctly
    assert preprocessor.data == sample_data

    # Check if the imputer is created correctly
    expected_imputer = {
        "numeric_col": 3.0,  # Median of [1, 2, 4, 5]
        "categorical_col": "a",  # Mode of ['a', 'b', 'a', 'b']
        "fold": 1.0,
        "target": 0.0,
    }
    assert preprocessor.imputer == expected_imputer


def test_default_preprocessor_data_getter(sample_data):
    preprocessor = DefaultPreprocessor(sample_data)

    # Check if the data getter returns the correct BaseModelData object
    assert preprocessor.data == sample_data


def test_default_preprocessor_data_setter(sample_data):
    preprocessor = DefaultPreprocessor(sample_data)

    # Create a new sample data
    new_data = {
        "numeric_col": [10, 20, 30, 40, 50],
        "categorical_col": ["x", "y", "z", "x", "y"],
        "fold": [1, 1, 1, 1, 1],
        "target": [1, 0, 1, 0, 1],
    }
    new_df = pd.DataFrame(new_data)
    new_sample_data = DefaultModelData(new_df)

    # Set the new data
    preprocessor.data = new_sample_data

    # Check if the data setter updates the BaseModelData object correctly
    assert preprocessor.data == new_sample_data


def test_default_preprocessor_df_getter(sample_data):
    preprocessor = DefaultPreprocessor(sample_data)

    # Check if the df getter returns the correct DataFrame
    assert preprocessor.df.equals(sample_data.df)


def test_default_preprocessor_new(sample_data):
    preprocessor = DefaultPreprocessor(sample_data)
    new_preprocessor = preprocessor.new

    # Check if the new instance is a DefaultPreprocessor
    assert isinstance(new_preprocessor, DefaultPreprocessor)

    # Check if the new instance has the same data as the original
    assert new_preprocessor.data == preprocessor.data

    # Check if the new instance has a different imputer object (not the same reference)
    assert new_preprocessor.imputer == preprocessor.imputer
    assert new_preprocessor.imputer is not preprocessor.imputer


def test_default_preprocessor_imputer(sample_data):
    preprocessor = DefaultPreprocessor(sample_data)

    # Check if the imputer is created correctly
    expected_imputer = {
        "numeric_col": 3.0,  # Median of [1, 2, 4, 5]
        "categorical_col": "a",  # Mode of ['a', 'b', 'a', 'b']
        "fold": 1.0,
        "target": 0.0,
    }
    assert preprocessor.imputer == expected_imputer

    # Modify the data and check if the imputer updates correctly
    new_data = {
        "numeric_col": [10, 20, None, 40, 50],
        "categorical_col": ["x", "y", None, "x", "y"],
        "fold": [1, 1, 1, 1, 1],
        "target": [1, 0, 1, 0, 1],
    }
    new_df = pd.DataFrame(new_data)
    new_sample_data = DefaultModelData(new_df)
    preprocessor.data = new_sample_data

    expected_new_imputer = {
        "numeric_col": 30.0,  # Median of [10, 20, 40, 50]
        "categorical_col": "x",  # Mode of ['x', 'y', 'x', 'y']
        "fold": 1.0,
        "target": 1.0,
    }
    assert preprocessor.imputer == expected_new_imputer


def test_get_missing_value_imputer(sample_data):
    preprocessor = DefaultPreprocessor(sample_data)
    df = sample_data.df

    # Test imputer for the initial sample data
    expected_imputer = {
        "numeric_col": 3.0,  # Median of [1, 2, 4, 5]
        "categorical_col": "a",  # Mode of ['a', 'b', 'a', 'b']
        "fold": 1.0,
        "target": 0.0,
    }
    assert preprocessor._get_missing_value_imputer(df) == expected_imputer  # noqa: SLF001


def test_imputer_for_data_without_missing_values():
    no_missing_data = {
        "numeric_col": [10, 20, 30, 40, 50],
        "categorical_col": ["x", "y", "z", "x", "y"],
        "fold": [1, 1, 1, 1, 1],
        "target": [1, 0, 1, 0, 1],
    }
    no_missing_df = pd.DataFrame(no_missing_data)
    no_missing_sample_data = DefaultModelData(no_missing_df)
    expected_no_missing_imputer = {
        "numeric_col": 30.0,  # Median of [10, 20, 30, 40, 50]
        "categorical_col": "x",  # Mode of ['x', 'y', 'z', 'x', 'y']
        "fold": 1.0,
        "target": 1.0,
    }

    preprocessor = DefaultPreprocessor(no_missing_sample_data)
    assert (
        preprocessor._get_missing_value_imputer(no_missing_df)  # noqa: SLF001
        == expected_no_missing_imputer
    )


def test_imputer_for_data_with_only_missing_values():
    # Test imputer for data with all missing values in a column
    all_missing_data = {
        "numeric_col": [None, None, None, None, None],
        "categorical_col": [None, None, None, None, None],
        "fold": [None, None, None, None, None],
        "target": [None, None, None, None, None],
    }

    expected_all_missing_imputer = {
        "numeric_col": None,
        "categorical_col": None,
        "fold": None,
        "target": None,
    }

    all_missing_df = pd.DataFrame(all_missing_data)

    preprocessor = DefaultPreprocessor(DefaultModelData(all_missing_df))

    assert (
        preprocessor._get_missing_value_imputer(all_missing_df)  # noqa: SLF001
        == expected_all_missing_imputer
    )


def test_imputer_for_data_with_mixed_missing_values():
    mixed_missing_data = {
        "numeric_col": [None, 20, None, 40, None],
        "categorical_col": [None, "y", None, "x", None],
        "fold": [None, 1, None, 1, None],
        "target": [None, 0, None, 1, None],
    }
    mixed_missing_df = pd.DataFrame(mixed_missing_data)
    mixed_missing_sample_data = DefaultModelData(mixed_missing_df)
    expected_mixed_missing_imputer = {
        "numeric_col": 30.0,  # Median of [20, 40]
        "categorical_col": "x",  # Mode of ['y', 'x']
        "fold": 1.0,
        "target": 0.5,  # Median of [0, 1]
    }
    preprocessor = DefaultPreprocessor(mixed_missing_sample_data)
    assert (
        preprocessor._get_missing_value_imputer(mixed_missing_df)  # noqa: SLF001
        == expected_mixed_missing_imputer
    )


def test_handle_missing_values(sample_data):
    preprocessor = DefaultPreprocessor(sample_data)
    preprocessor.handle_missing_values()

    # Check if missing values are handled correctly
    expected_df = pd.DataFrame(
        {
            "numeric_col": [1, 2, 3.0, 4, 5],  # 3.0 is the median
            "categorical_col": ["a", "b", "a", "a", "b"],  # 'a' is the mode
            "fold": [0, 1, 2, 2, 1],
            "target": [0, 1, 0, 1, 0],
        }
    )

    pd.testing.assert_frame_equal(preprocessor.df, expected_df)


def test_handle_missing_values_with_no_missing_data():
    no_missing_data = {
        "numeric_col": [10, 20, 30, 40, 50],
        "categorical_col": ["x", "y", "z", "x", "y"],
        "fold": [1, 1, 1, 1, 1],
        "target": [1, 0, 1, 0, 1],
    }
    no_missing_df = pd.DataFrame(no_missing_data)
    no_missing_sample_data = DefaultModelData(no_missing_df)

    preprocessor = DefaultPreprocessor(no_missing_sample_data)
    preprocessor.handle_missing_values()

    # Check if the DataFrame remains unchanged
    pd.testing.assert_frame_equal(preprocessor.df, no_missing_df)


def test_handle_missing_values_with_all_missing_data():
    all_missing_data = {
        "numeric_col": [None, None, None, None, None],
        "categorical_col": [None, None, None, None, None],
        "fold": [None, None, None, None, None],
        "target": [None, None, None, None, None],
    }
    all_missing_df = pd.DataFrame(all_missing_data)
    all_missing_sample_data = DefaultModelData(all_missing_df)

    preprocessor = DefaultPreprocessor(all_missing_sample_data)
    preprocessor.handle_missing_values()

    # Check if the DataFrame remains unchanged (since all values are missing)
    pd.testing.assert_frame_equal(preprocessor.df, all_missing_df)


def test_handle_missing_values_with_mixed_missing_data():
    mixed_missing_data = {
        "numeric_col": [None, 20, None, 40, None],
        "categorical_col": [None, "y", None, "x", None],
        "fold": [None, 1, None, 1, None],
        "target": [None, 0, None, 1, None],
    }
    mixed_missing_df = pd.DataFrame(mixed_missing_data)
    mixed_missing_sample_data = DefaultModelData(mixed_missing_df)

    preprocessor = DefaultPreprocessor(mixed_missing_sample_data)
    preprocessor.handle_missing_values()

    expected_df = pd.DataFrame(
        {
            "numeric_col": [30.0, 20, 30.0, 40, 30.0],  # 30.0 is the median
            "categorical_col": ["x", "y", "x", "x", "x"],  # 'x' is the mode
            "fold": [1.0, 1, 1.0, 1, 1.0],
            "target": [0.5, 0, 0.5, 1, 0.5],  # 0.5 is the median
        }
    )

    pd.testing.assert_frame_equal(preprocessor.df, expected_df)


def test_run_with_missing_values(sample_data):
    preprocessor = DefaultPreprocessor(sample_data)
    result_df = preprocessor.run()

    # Check if missing values are handled correctly
    expected_df = pd.DataFrame(
        {
            "numeric_col": [1, 2, 3.0, 4, 5],  # 3.0 is the median
            "categorical_col": ["a", "b", "a", "a", "b"],  # 'a' is the mode
            "fold": [0, 1, 2, 2, 1],
            "target": [0, 1, 0, 1, 0],
        }
    )

    pd.testing.assert_frame_equal(result_df, expected_df)


def test_run_with_no_missing_values():
    no_missing_data = {
        "numeric_col": [10, 20, 30, 40, 50],
        "categorical_col": ["x", "y", "z", "x", "y"],
        "fold": [1, 1, 1, 1, 1],
        "target": [1, 0, 1, 0, 1],
    }
    no_missing_df = pd.DataFrame(no_missing_data)
    no_missing_sample_data = DefaultModelData(no_missing_df)

    preprocessor = DefaultPreprocessor(no_missing_sample_data)
    result_df = preprocessor.run()

    # Check if the DataFrame remains unchanged
    pd.testing.assert_frame_equal(result_df, no_missing_df)


def test_run_with_all_missing_values():
    all_missing_data = {
        "numeric_col": [None, None, None, None, None],
        "categorical_col": [None, None, None, None, None],
        "fold": [None, None, None, None, None],
        "target": [None, None, None, None, None],
    }
    all_missing_df = pd.DataFrame(all_missing_data)
    all_missing_sample_data = DefaultModelData(all_missing_df)

    preprocessor = DefaultPreprocessor(all_missing_sample_data)
    result_df = preprocessor.run()

    # Check if the DataFrame remains unchanged (since all values are missing)
    pd.testing.assert_frame_equal(result_df, all_missing_df)


def test_run_with_mixed_missing_values():
    mixed_missing_data = {
        "numeric_col": [None, 20, None, 40, None],
        "categorical_col": [None, "y", None, "x", None],
        "fold": [None, 1, None, 1, None],
        "target": [None, 0, None, 1, None],
    }
    mixed_missing_df = pd.DataFrame(mixed_missing_data)
    mixed_missing_sample_data = DefaultModelData(mixed_missing_df)

    preprocessor = DefaultPreprocessor(mixed_missing_sample_data)
    result_df = preprocessor.run()

    expected_df = pd.DataFrame(
        {
            "numeric_col": [30.0, 20, 30.0, 40, 30.0],  # 30.0 is the median
            "categorical_col": ["x", "y", "x", "x", "x"],  # 'x' is the mode
            "fold": [1.0, 1, 1.0, 1, 1.0],
            "target": [0.5, 0, 0.5, 1, 0.5],  # 0.5 is the median
        }
    )

    pd.testing.assert_frame_equal(result_df, expected_df)
