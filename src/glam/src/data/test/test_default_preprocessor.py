"""Test functionality with unit tests for the DefaultPreprocessor class."""

from __future__ import annotations

import re
from typing import Literal
import pytest
import pandas as pd
import numpy as np
from glam.src.data.default_model_data import DefaultModelData
from glam.src.data.data_prep.preprocessors.default_preprocessor import (
    DefaultPreprocessor,
)


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """Fixture to create a sample DataFrame for testing."""
    return pd.DataFrame(
        {
            "y": [1, 2, 3, 4, 5],
            "cv": [1, 2, 3, 4, 5],
            "numeric_feature": [1, 2, np.nan, 4, 5],
            "categorical_feature": ["A", "B", np.nan, "A", "C"],
            "all_missing": [np.nan, np.nan, np.nan, np.nan, np.nan],
            "single_value": [1, 1, 1, 1, 1],
        }
    )


@pytest.fixture
def base_model_data(sample_dataframe: pd.DataFrame) -> DefaultModelData:
    """Fixture to create a BaseModelData instance."""
    return DefaultModelData(sample_dataframe, "y", "cv")


@pytest.fixture
def preprocessor(base_model_data: DefaultModelData) -> DefaultPreprocessor:
    """Fixture to create a DefaultPreprocessor instance."""
    return DefaultPreprocessor(base_model_data)


def test_imputer_creation(preprocessor: DefaultPreprocessor) -> None:
    """Test if the imputer is correctly created for each column."""
    expected_imputer = {
        "numeric_feature": 3.0,  # Median of [1, 2, 4, 5]
        "categorical_feature": "A",  # Mode of ["A", "B", "A", "C"]
        "all_missing": np.nan,  # No non-missing value, should remain NaN
        "single_value": 1,  # Only one value available
    }
    assert (
        preprocessor.imputer["numeric_feature"] == expected_imputer["numeric_feature"]
    )
    assert (
        preprocessor.imputer["categorical_feature"]
        == expected_imputer["categorical_feature"]
    )
    assert np.isnan(preprocessor.imputer["all_missing"])
    assert preprocessor.imputer["single_value"] == expected_imputer["single_value"]


@pytest.mark.parametrize(
    "column_name,expected_value",
    [
        ("numeric_feature", [1, 2, 3.0, 4, 5]),  # NaN replaced with median 3.0
        (
            "categorical_feature",
            ["A", "B", "A", "A", "C"],
        ),  # NaN replaced with mode 'A'
        ("all_missing", [np.nan, np.nan, np.nan, np.nan, np.nan]),  # Should remain NaN
        ("single_value", [1, 1, 1, 1, 1]),  # No change
    ],
)
def test_handle_missing_values(
    preprocessor: DefaultPreprocessor,
    column_name: Literal[
        "numeric_feature", "categorical_feature", "all_missing", "single_value"
    ],
    expected_value: list[int | float] | list[str] | list[float] | list[int],
) -> None:
    """Test if handle_missing_values correctly imputes missing data."""
    preprocessor.handle_missing_values()
    result = preprocessor.df[column_name].tolist()

    # replace nans with -123456789 to compare with np.testing.assert_array_equal
    if isinstance(expected_value[0], str):
        expected_value = ["-123456789" if x is np.nan else x for x in expected_value]
        result = ["-123456789" if x is np.nan else x for x in result]
    elif isinstance(expected_value[0], (float, int)):
        expected_value = [-123456789 if np.isnan(x) else x for x in expected_value]
        result = [-123456789 if np.isnan(x) else x for x in result]
    else:
        raise ValueError(
            f"Unexpected data type in expected_value: {expected_value[0]} is of type {type(expected_value[0])}"
        )

    np.testing.assert_array_equal(result, expected_value)


def test_run_method(preprocessor: DefaultPreprocessor) -> None:
    """Test the run method executes all preprocessing steps."""
    result_df = preprocessor.run()

    # Check the results after running the full preprocessing
    expected_numeric = [1, 2, 3.0, 4, 5]
    expected_categorical = ["A", "B", "A", "A", "C"]

    assert result_df["numeric_feature"].tolist() == expected_numeric
    assert result_df["categorical_feature"].tolist() == expected_categorical


def test_new_preprocessor_instance(preprocessor: DefaultPreprocessor) -> None:
    """Test the 'new' property creates a new instance of the preprocessor."""
    new_preprocessor = preprocessor.new
    assert isinstance(new_preprocessor, DefaultPreprocessor)
    assert new_preprocessor is not preprocessor  # Ensure it's a new instance


def test_data_setter_getter(
    preprocessor: DefaultPreprocessor, base_model_data: DefaultModelData
) -> None:
    """Test the data property getter and setter."""
    new_df = base_model_data.df.copy()
    new_df["new_column"] = [10, 20, 30, 40, 50]

    new_data = DefaultModelData(new_df, "y", "cv")
    preprocessor.data = new_data

    assert preprocessor.data.df.equals(new_df)


@pytest.mark.parametrize("col_name", ["numeric_feature", "categorical_feature"])
def test_imputer_missing_values_in_a_column(
    preprocessor: DefaultPreprocessor,
    col_name: Literal["numeric_feature", "categorical_feature"],
) -> None:
    """Test the imputer for special cases like all missing values or unusual edge cases."""
    df = preprocessor.df
    df[col_name] = np.nan if col_name == "numeric_feature" else ""
    imputer = preprocessor._get_missing_value_imputer(df)  # noqa: SLF001 ## this is ok -- setting up an abnormal case

    # For all missing numeric, median should be NaN
    # For all missing categorical, it should fallback to an empty string or specific edge handling logic
    if col_name == "numeric_feature":
        assert np.isnan(imputer[col_name])
    else:
        assert imputer[col_name] == ""
