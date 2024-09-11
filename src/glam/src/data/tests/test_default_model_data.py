import re
import pytest
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal
from glam.src.data.default_model_data import DefaultModelData


@pytest.fixture
def test_df():
    data = {
        "feature1": [1, 2, 3],
        "feature2": [4, 5, 6],
        "fold": [1, 1, 2],
        "target": [0, 1, 0],
    }
    return pd.DataFrame(data)


@pytest.fixture
def test_data(test_df):
    return DefaultModelData(test_df, y="target", cv="fold")


@pytest.mark.parametrize(
    "y, cv, unanalyzed",
    [(None, None, None), ("target", "fold", []), ("target", "fold", ["feature1"])],
)
def test_initialization(y, cv, unanalyzed):
    data = {
        "feature1": [1, 2, 3],
        "feature2": [4, 5, 6],
        "fold": [1, 1, 2],
        "target": [0, 1, 0],
    }
    df = pd.DataFrame(data)
    model_data = DefaultModelData(df, y=y, cv=cv, unanalyzed=unanalyzed)

    assert (
        model_data._y == (y if y is not None else "target")  # noqa: SLF001
    ), f"Expected y attribute to be '{y if y is not None else 'target'}', got '{model_data._y}'"  # noqa: SLF001
    assert (
        model_data._cv == (cv if cv is not None else "fold")  # noqa: SLF001
    ), f"Expected cv attribute to be '{cv if cv is not None else 'fold'}', got '{model_data._cv}'"  # noqa: SLF001
    assert (
        model_data._unanalyzed == (unanalyzed if unanalyzed is not None else [])  # noqa: SLF001
    ), f"Expected unanalyzed to be '{unanalyzed if unanalyzed is not None else []}', got '{model_data._unanalyzed}'"  # noqa: SLF001
    assert model_data.is_time_series_cv is True, "Expected is_time_series_cv to be True"


@pytest.fixture
def sample_data():
    data = {
        "feature1": [1, 2, 3],
        "feature2": [4, 5, 6],
        "fold": [1, 1, 2],
        "target": [0, 1, 0],
    }
    df = pd.DataFrame(data)
    return DefaultModelData(df, y="target", cv="fold")


def test_df_getter_setter(sample_data):
    new_data = {
        "feature1": [10, 20, 30],
        "feature2": [40, 50, 60],
        "fold": [2, 2, 3],
        "target": [0, 0, 1],
    }
    new_df = pd.DataFrame(new_data)
    sample_data.df = new_df
    assert_frame_equal(
        sample_data.df, new_df, "DataFrame setter or getter not working as expected"
    )


def test_X(sample_data):
    expected_columns = ["feature1", "feature2"]
    assert set(sample_data.X.columns.tolist()) == set(
        expected_columns
    ), f"Expected X to have columns {expected_columns}, got {sample_data.X.columns}"


def test_y(sample_data):
    assert_series_equal(
        sample_data.y,
        sample_data.df["target"],
        "y property not returning expected Series",
    )


def test_cv(sample_data):
    assert_series_equal(
        sample_data.cv,
        sample_data.df["fold"],
        "cv property not returning expected Series",
    )


def test_add_feature(sample_data):
    new_feature_values = pd.Series([7, 8, 9], name="new_feature")
    sample_data.add_feature("new_feature", new_feature_values)
    assert (
        "new_feature" in sample_data.df_cols
    ), "New feature 'new_feature' not added to DataFrame"
    assert_series_equal(
        sample_data.df["new_feature"],
        new_feature_values,
        "New feature values do not match expected series",
    )


def test_empty_dataframe_initialization():
    df = pd.DataFrame()
    with pytest.raises(ValueError, match="DataFrame cannot be empty"):
        DefaultModelData(df)


def test_missing_columns():
    df = pd.DataFrame({"feature1": [1, 2, 3]})
    with pytest.raises(
        KeyError, match="Response variable 'target' not found in the DataFrame"
    ):
        DefaultModelData(df, y="target", cv="fold")


def test_y_initialized_correctly__y_included(test_data):
    assert_series_equal(test_data.y, pd.Series([0, 1, 0], name="target"))


def test_y_initialized_correctly__y_not_included(test_df):
    model_data = DefaultModelData(test_df, y=None, cv="fold")
    assert_series_equal(model_data.y, pd.Series([0, 1, 0], name="target"))


def test_y_not_in_df_columns(test_df):
    with pytest.raises(
        KeyError, match="Response variable 'target' not found in the DataFrame"
    ):
        DefaultModelData(test_df.drop(columns="target"), y="target", cv="fold")

    with pytest.raises(
        KeyError, match="Response variable 'None' not found in the DataFrame"
    ):
        DefaultModelData(test_df.drop(columns="target"), y=None, cv="fold")


def test_cv_initialized_correctly__cv_included(test_data):
    assert_series_equal(test_data.cv, pd.Series([1, 1, 2], name="fold"))


def test_cv_initialized_correctly__cv_not_included(test_df):
    data = DefaultModelData(test_df, y="target", cv=None)
    assert_series_equal(data.cv, pd.Series([1, 1, 2], name="fold"))


def test_cv_not_in_df_columns(test_df):
    with pytest.raises(
        KeyError, match="Cross-validation fold column 'fold' not found in the DataFrame"
    ):
        DefaultModelData(test_df.drop(columns=["fold"]), y="target", cv="fold")

    with pytest.raises(
        KeyError, match="Cross-validation fold column 'None' not found in the DataFrame"
    ):
        DefaultModelData(test_df.drop(columns=["fold"]), y="target", cv=None)


def test_repr(test_data):
    assert repr(test_data) == "ModelData(y='target', cv='fold', df.shape=(3, 4))"


def test_str(test_data):
    assert str(test_data) == "ModelData(y='target', cv='fold', df.shape=(3, 4))"


def test_df_property(test_data):
    expected_df = pd.DataFrame(
        {
            "feature1": [1, 2, 3],
            "feature2": [4, 5, 6],
            "fold": [1, 1, 2],
            "target": [0, 1, 0],
        }
    )
    assert_frame_equal(test_data.df, expected_df)


def test_df_shape(test_data):
    assert test_data.df.shape == (3, 4)


def test_unanalyzed_getter(test_data):
    assert (
        test_data.unanalyzed == []
    ), "Expected unanalyzed to be an empty list by default"


def test_unanalyzed_setter(test_data):
    new_unanalyzed = ["feature1", "feature2"]
    test_data.unanalyzed = new_unanalyzed
    assert (
        test_data.unanalyzed == new_unanalyzed
    ), f"Expected unanalyzed to be {new_unanalyzed}, got {test_data.unanalyzed}"


def test_is_time_series_cv_default(test_data):
    assert (
        test_data.is_time_series_cv is True
    ), "Expected is_time_series_cv to be True by default"


def test_is_time_series_cv_false():
    data = {
        "feature1": [1, 2, 3],
        "feature2": [4, 5, 6],
        "fold": [1, 1, 2],
        "target": [0, 1, 0],
    }
    df = pd.DataFrame(data)
    model_data = DefaultModelData(df, y="target", cv="fold", is_time_series_cv=False)
    assert (
        model_data.is_time_series_cv is False
    ), "Expected is_time_series_cv to be False"


def test_add_feature3(sample_data):
    # Test adding a new feature
    new_feature_values = pd.Series([7, 8, 9], name="new_feature")
    sample_data.add_feature("new_feature", new_feature_values)
    assert (
        "new_feature" in sample_data.df.columns
    ), "New feature 'new_feature' not added to DataFrame"
    assert_series_equal(
        sample_data.df["new_feature"],
        new_feature_values,
        "New feature values do not match expected series",
    )

    # Test adding a feature with existing name
    with pytest.raises(
        ValueError, match="Feature 'new_feature' already exists in the DataFrame"
    ):
        sample_data.add_feature("new_feature", new_feature_values)

    # Test adding a feature with mismatched length
    mismatched_values = pd.Series([7, 8], name="mismatched_feature")
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Length of new feature 'mismatched_feature' (2) does not match the number of rows in the DataFrame (3)"
        ),
    ):
        sample_data.add_feature("mismatched_feature", mismatched_values)
