"""Test functionality with unit tests for the DefaultModelData class."""

import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal
import pytest
from glam.src.data.default_model_data import DefaultModelData

DATAFRAME = pd.DataFrame(
    {"y": [0, 1, 0], "cv": [0, 0, 1], "A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]}
)


class TestModelData:
    """Test the DefaultModelData class."""

    @property
    def model_data(self) -> DefaultModelData:
        """Return a DefaultModelData object."""
        return self._model_data

    @property
    def df(self) -> pd.DataFrame:
        """Return a sample DataFrame."""
        return self._model_data.df

    def setup_method(self) -> None:
        """Set up method for the test case."""
        self._model_data = DefaultModelData(DATAFRAME, y="y", cv="cv")

    def test_add_feature(self) -> None:
        """Test if the add_feature method adds a new feature to the DataFrame."""
        # Add a new feature
        new_feature = pd.Series([10, 11, 12])
        self.model_data.add_feature("D", new_feature)

        # Create the expected DataFrame
        expected_df = pd.DataFrame(
            {
                "y": [0, 1, 0],
                "cv": [0, 0, 1],
                "A": [1, 2, 3],
                "B": [4, 5, 6],
                "C": [7, 8, 9],
                "D": [10, 11, 12],
            }
        )

        # Check if the new feature is added to the DataFrame
        assert_frame_equal(self.model_data.df, expected_df)
        assert_series_equal(self.df["D"], new_feature, check_names=False)

    pytest.main()
