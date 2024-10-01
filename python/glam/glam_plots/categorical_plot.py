"""Categorical plot."""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from glam.src.glam_plots.utils import get_marker_props, get_line_props


def categorical_plot(model, data, test_feature: str = "business_category") -> go.Figure:
    """Create a categorical plot."""
    model2 = model.copy()
    model2.add_feature(test_feature)
    model2.fit()

    df = pd.concat([model.y, model.cv, model.X], axis=1)
    df = df.loc[df["fold"] >= 10]
    df[test_feature] = data.df[test_feature]
    df["current_model"] = model.yhat_proba(
        df.drop(columns=[model.y.name, model.cv.name, test_feature])
    )
    df["test_model"] = model2.yhat_proba(df.drop(columns=[model.y.name, model.cv.name]))

    count_by_level = (
        df.groupby(test_feature)
        .count()[["hit_count"]]
        .reset_index()
        .rename(columns={"hit_count": "count"})
    )

    ave_by_level = (
        df.groupby(test_feature)
        .mean()[["hit_count", "current_model", "test_model"]]
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
            y=ave_by_level["current_model"],
            mode="lines+markers",
            name="Current Model",
            marker=get_marker_props("red"),
            line=get_line_props("red", 1.5, "dashdot"),
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scattergl(
            x=ave_by_level[test_feature],
            y=ave_by_level["test_model"],
            mode="lines+markers",
            name="Test Model",
            marker=get_marker_props("green", size=7),
            line=get_line_props("green", 1.5, "dashdot"),
        ),
        secondary_y=False,
    )

    fig.update_layout(
        title="Actual vs. Model Predictions",
        xaxis_title=test_feature,
        yaxis_title="Hit Ratio",
        showlegend=True,
    )

    fig.update_yaxes(title_text="Number of Quotes", secondary_y=True)

    return fig
