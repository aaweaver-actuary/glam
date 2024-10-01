import pytest
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from glam.src.data.default_model_data import DefaultModelData
from glam.src.fitters.statsmodels_formula_glm_fitter import StatsmodelsFormulaGlmFitter
from glam.src.fitted_model.statsmodels_fitted_glm import StatsmodelsFittedGlm
from glam.plots import (
    BasePlotConfig,
    get_configs,
    create_plot,
    create_ave_by_level_data,
    create_test_model,
    create_pre_binning_data,
    numeric_plot,
    categorical_plot,
    _add_bar,
    _add_scatter,
    _update_layout,
)


@pytest.fixture
def sample_dataframe():
    """Fixture to create a sample DataFrame for testing."""
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "feature": rng.normal(size=100),
            "hit_count": rng.integers(0, 2, 100),
            "current_model": rng.beta(2, 2, 100),
            "test_model": rng.beta(2, 2, 100),
            "count": rng.integers(10, 100, 100),
        }
    )


@pytest.fixture
def default_model_data(sample_dataframe):
    """Fixture to create a DefaultModelData instance."""
    return DefaultModelData(df=sample_dataframe, y="hit_count", cv="count")


@pytest.fixture
def fitted_model(default_model_data) -> StatsmodelsFittedGlm:
    """Fixture to create a StatsmodelsFittedGlm instance."""
    formula = "hit_count ~ feature"
    fitter = StatsmodelsFormulaGlmFitter()
    return fitter.fit(formula, default_model_data.X, default_model_data.y)


def test_get_configs():
    """Test the get_configs function."""
    configs = get_configs()
    assert len(configs) == 4, f"Expected 4 configs, got {len(configs)}"

    for config in configs:
        assert isinstance(
            config, BasePlotConfig
        ), f"Expected BasePlotConfig, got {type(config)}"


def test_add_bar(sample_dataframe):
    """Test the _add_bar function."""
    fig = go.Figure()
    config = get_configs()[3]  # Use the CountConfig
    fig = _add_bar(fig, sample_dataframe["feature"], sample_dataframe["count"], config)
    assert len(fig.data) == 1, f"Expected 1 trace, got {len(fig.data)}"
    assert isinstance(
        fig.data[0], go.Bar
    ), f"Expected Bar trace, got {type(fig.data[0])}"
    assert (
        fig.data[0].name == config.name
    ), f"Expected name {config.name} to be the name, got {fig.data[0].name}"


def test_add_scatter(sample_dataframe):
    """Test the _add_scatter function."""
    fig = go.Figure()
    config = get_configs()[2]  # Use the ActualConfig
    fig = _add_scatter(
        fig, sample_dataframe["feature"], sample_dataframe["hit_count"], config
    )
    assert len(fig.data) == 1, f"Expected 1 trace, got {len(fig.data)}"
    assert isinstance(
        fig.data[0], go.Scattergl
    ), f"Expected Scatter trace, got {type(fig.data[0])}"
    assert (
        fig.data[0].name == config.name
    ), f"Expected name {config.name} to be the name, got {fig.data[0].name}"


def test_update_layout():
    """Test the _update_layout function."""
    fig = go.Figure()
    fig = _update_layout(fig, "Test Title", "X Axis", "Y Axis", "Y2 Axis")
    layout = fig.layout
    assert (
        layout.title.text == "Test Title"
    ), f"Expected title 'Test Title', got {layout.title.text}"
    assert (
        layout.xaxis.title.text == "X Axis"
    ), f"Expected X Axis, got {layout.xaxis.title.text}"
    assert (
        layout.yaxis.title.text == "Y Axis"
    ), f"Expected Y Axis, got {layout.yaxis.title.text}"


def test_create_plot(sample_dataframe):
    """Test the create_plot function."""
    fig = create_plot(sample_dataframe, "feature")
    assert isinstance(fig, go.Figure), f"Expected go.Figure, got {type(fig)}"
    # 1 bar and 3 scatter traces
    assert len(fig.data) == 4, f"Expected 4 traces, got {len(fig.data)}"
    assert (
        fig.layout.title.text == "Actual vs. Model Predictions"
    ), f"Expected 'Actual vs. Model Predictions', got {fig.layout.title.text}"


def test_create_ave_by_level_data(sample_dataframe):
    """Test the create_ave_by_level_data function."""
    result = create_ave_by_level_data(sample_dataframe, "feature")
    assert isinstance(
        result, pd.DataFrame
    ), f"Expected pd.DataFrame, got {type(result)}"
    assert "hit_count" in result.columns, f"hit_count not in columns:\n{result.columns}"
    assert "count" in result.columns, f"count not in columns:\n{result.columns}"
    assert result.index.name == "feature", f"Expected feature, got {result.index.name}"


def test_create_test_model(fitted_model):
    """Test the create_test_model function."""
    new_model = create_test_model(fitted_model, "feature")
    assert isinstance(
        new_model, StatsmodelsFittedGlm
    ), f"Expected StatsmodelsFittedGlm, got {type(new_model)}"


def test_create_pre_binning_data(default_model_data, fitted_model):
    """Test the create_pre_binning_data function."""
    df = create_pre_binning_data(default_model_data, fitted_model, "feature")
    assert isinstance(df, pd.DataFrame), f"Expected pd.DataFrame, got {type(df)}"
    assert "current_model" in df.columns, f"current_model not in columns:\n{df.columns}"
    assert "test_model" in df.columns, f"test_model not in columns:\n{df.columns}"


def test_numeric_plot(default_model_data, fitted_model):
    """Test the numeric_plot function."""
    fig = numeric_plot(default_model_data, fitted_model, "feature", n_bins=5)
    assert isinstance(fig, go.Figure), f"Expected go.Figure, got {type(fig)}"

    # 1 bar and 3 scatter traces
    assert len(fig.data) == 4, f"Expected 4 traces, got {len(fig.data)}"


def test_categorical_plot(default_model_data, fitted_model):
    """Test the categorical_plot function."""
    fig = categorical_plot(default_model_data, fitted_model, "feature")
    assert isinstance(fig, go.Figure), f"Expected go.Figure, got {type(fig)}"

    # 1 bar and 3 scatter traces
    assert len(fig.data) == 4, f"Expected 4 traces, got {len(fig.data)}"
