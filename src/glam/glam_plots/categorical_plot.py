"""Categorical plot."""

import plotly.graph_objects as go  # type: ignore
from plotly.subplots import make_subplots  # type: ignore
import pandas as pd
from glam.src.glam_plots.utils import get_marker_props, get_line_props  # type: ignore
from glam.analysis.base_analysis import BaseAnalysis
from glam.src.data.base_model_data import BaseModelData


def categorical_plot(
    analysis: BaseAnalysis, data: BaseModelData, test_feature: str = "business_category"
) -> go.Figure:
    """Create a categorical plot."""
    analysis2 = analysis.copy()
    analysis2.add_feature(test_feature)
    analysis2.fit()

    df = pd.concat([analysis.y, analysis.cv, analysis.X], axis=1)
    df = df.loc[df["fold"] >= 10]
    df[test_feature] = data.df[test_feature]
    df["current_analysis"] = analysis.yhat_proba(
        df.drop(columns=[analysis.y.name, analysis.cv.name, test_feature])
    )
    df["test_analysis"] = analysis2.yhat_proba(
        df.drop(columns=[analysis.y.name, analysis.cv.name])
    )

    count_by_level = (
        df.groupby(test_feature)
        .count()[["hit_count"]]
        .reset_index()
        .rename(columns={"hit_count": "count"})
    )

    ave_by_level = (
        df.groupby(test_feature)
        .mean()[["hit_count", "current_analysis", "test_analysis"]]
        .reset_index()
        .set_index(test_feature)
        .join(count_by_level.set_index(test_feature), on=test_feature)
        .reset_index()
        .sort_values(test_feature, ascending=True)
    )

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Bar(
            x=ave_by_level[test_feature],
            y=ave_by_level["count"],
            name="Count",
            marker=get_marker_props("grey", 0.5, 0.5),
            showlegend=False,
        ),
        secondary_y=True,
    )

    fig.add_trace(
        go.Scattergl(
            x=ave_by_level[test_feature],
            y=ave_by_level["hit_count"],
            mode="lines+markers",
            name="Actual",
            marker=get_marker_props("blue"),
            line=get_line_props("blue"),
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scattergl(
            x=ave_by_level[test_feature],
            y=ave_by_level["current_analysis"],
            mode="lines+markers",
            name="Current analysis",
            marker=get_marker_props("red"),
            line=get_line_props("red", 1.5, "dashdot"),
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scattergl(
            x=ave_by_level[test_feature],
            y=ave_by_level["test_analysis"],
            mode="lines+markers",
            name="Test analysis",
            marker=get_marker_props("green", size=7),
            line=get_line_props("green", 1.5, "dashdot"),
        ),
        secondary_y=False,
    )

    fig.update_layout(
        title="Actual vs. analysis Predictions",
        xaxis_title=test_feature,
        yaxis_title="Hit Ratio",
        showlegend=True,
    )

    fig.update_yaxes(title_text="Number of Quotes", secondary_y=True)

    return fig
