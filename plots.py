"""Plotting functions for the GLAM package."""

from __future__ import annotations
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from glam.src.data.base_model_data import BaseModelData
from glam.src.fitted_model.base_fitted_model import BaseFittedModel

from dataclasses import dataclass


@dataclass
class BasePlotConfig:
    """Base class for plot configurations."""

    name: str = "NO_NAME"
    color: str = "gray"
    marker__size: int = 10
    marker__opacity: float = 0.5
    marker__line_color: str = "black"
    marker__line_width: float = 0.5
    line__width: float = 1.5
    line__dash: str = "dashdot"
    show_legend: bool = True
    secondary_y: bool = False
    mode: str = "lines+markers"


@dataclass
class CurrentModelConfig(BasePlotConfig):
    """Configuration for the current model plot."""

    name: str = "Current Model"
    color: str = "red"
    marker__size: int = 7


@dataclass
class TestModelConfig(BasePlotConfig):
    """Configuration for the test model plot."""

    name: str = "Test Model"
    color: str = "green"
    marker__size: int = 7


@dataclass
class ActualConfig(BasePlotConfig):
    """Configuration for the actual plot."""

    name: str = "Actual"
    color: str = "blue"
    marker__size: int = 10
    line__width: float = 2
    line__dash: str = "dashdot"


@dataclass
class CountConfig(BasePlotConfig):
    """Configuration for the count plot."""

    name: str = "Count"
    color: str = "gray"
    marker__size: int = 10
    marker__opacity: float = 0.5
    marker__line_color: str = "black"
    marker__line_width: float = 0.5


def get_configs() -> (
    tuple[BasePlotConfig, BasePlotConfig, BasePlotConfig, BasePlotConfig]
):
    """Return the configuration objects for the plot."""
    current_model_config = CurrentModelConfig()
    test_model_config = TestModelConfig()
    actual_config = ActualConfig()
    count_config = CountConfig()
    return current_model_config, test_model_config, actual_config, count_config


def _add_bar(
    fig: go.Figure, x: pd.Series, y: pd.Series, config: BasePlotConfig
) -> go.Figure:
    """Add a trace to the figure."""
    fig.add_trace(
        go.Bar(
            x=x,
            y=y,
            name=config.name,
            marker={
                "color": config.color,
                "opacity": config.marker__opacity,
                "line": {
                    "color": config.marker__line_color,
                    "width": config.marker__line_width,
                },
            },
            showlegend=config.show_legend,
        ),
        secondary_y=config.secondary_y,
    )
    return fig


def _add_scatter(
    fig: go.Figure, x: pd.Series, y: pd.Series, config: BasePlotConfig
) -> go.Figure:
    """Add a trace to the figure."""
    fig.add_trace(
        go.Scattergl(
            x=x,
            y=y,
            mode=config.mode,
            name=config.name,
            marker={
                "size": config.marker__size,
                "color": config.color,
                "opacity": config.marker__opacity,
                "line": {
                    "color": config.marker__line_color,
                    "width": config.marker__line_width,
                },
            },
            line={
                "color": config.color,
                "width": config.line__width,
                "dash": config.line__dash,
            },
            showlegend=config.show_legend,
        ),
        secondary_y=config.secondary_y,
    )
    return fig


def _update_layout(
    fig: go.Figure, title: str, xaxis_title: str, yaxis_title: str, yaxis2_title: str
) -> go.Figure:
    """Update the layout of the figure."""
    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        showlegend=True,
        secondary_y=False,
    )

    fig.update_yaxes(title_text=yaxis2_title, secondary_y=True)
    return fig


def create_plot(df: pd.DataFrame, feature_name: str) -> go.Figure:
    """Create a plot based on the feature type."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig = _add_bar(fig, df[feature_name], df["count"], CountConfig())
    fig = _add_scatter(fig, df[feature_name], df["hit_count"], ActualConfig())
    fig = _add_scatter(fig, df[feature_name], df["current_model"], CurrentModelConfig())
    fig = _add_scatter(fig, df[feature_name], df["test_model"], TestModelConfig())

    return _update_layout(
        fig,
        "Actual vs. Model Predictions",
        feature_name,
        "Hit Ratio",
        "Number of Quotes",
    )


def create_ave_by_level_data(df: pd.DataFrame, feature_name: str) -> pd.DataFrame:
    """Create a plot based on the feature type."""
    return (
        df.groupby(feature_name)
        .mean()[["hit_count", "current_model", "test_model"]]
        .reset_index()
        .set_index(feature_name)
        .join(
            df.groupby(feature_name)
            .count()[["hit_count"]]
            .reset_index()
            .rename(columns={"hit_count": "count"})
            .set_index(feature_name),
            on=feature_name,
        )
        .reset_index()
        .sort_values("hit_count", ascending=True)
    )


def create_test_model(model: BaseFittedModel, test_feature: str) -> BaseFittedModel:
    """Create a new model with the test feature added."""
    model2 = model.copy()
    model2.add_feature(test_feature)
    model2.fit()
    return model2


def create_pre_binning_data(
    data: BaseModelData, model: BaseFittedModel, test_feature: str
) -> pd.DataFrame:
    """Bin the numeric feature into n_bins and plot the actual hit ratio and model predictions."""
    model2 = create_test_model(model, test_feature)

    df = pd.concat([model.y, model.cv, model.X], axis=1)
    df = df.loc[df[model.cv.name] >= model.cv.max()]
    df[test_feature] = data.df[test_feature]
    df["current_model"] = model.yhat_proba(
        df.drop(columns=[model.y.name, model.cv.name, test_feature])
    )
    df["test_model"] = model2.yhat_proba(df.drop(columns=[model.y.name, model.cv.name]))

    return df


def numeric_plot(
    data: BaseModelData, model: BaseFittedModel, test_feature: str, n_bins: int = 5
) -> go.Figure:
    """Bin the numeric feature into n_bins and plot the actual hit ratio and model predictions."""
    df = create_pre_binning_data(data, model, test_feature)
    df["bin"] = pd.qcut(df[test_feature], n_bins, duplicates="drop").astype(str)
    plot_data = create_ave_by_level_data(df, "bin")
    return create_plot(plot_data, "bin")


def categorical_plot(
    data: BaseModelData, model: BaseFittedModel, test_feature: str
) -> go.Figure:
    """Plot the actual hit ratio and model predictions for a categorical feature."""
    df = create_pre_binning_data(data, model, test_feature)
    plot_data = create_ave_by_level_data(df, test_feature)
    return create_plot(plot_data, test_feature)
