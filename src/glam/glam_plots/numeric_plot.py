"""Create the numeric plot."""

import pandas as pd
import plotly.graph_objects as go  # type: ignore
from plotly.subplots import make_subplots  # type: ignore
from glam.src.glam_plots.utils import get_marker_props, get_line_props  # type: ignore
from glam.src.data.base_model_data import BaseModelData
from glam.analysis.base_glm_analysis import BaseGlmAnalysis


def numeric_plot(
    analysis: BaseGlmAnalysis, data: BaseModelData, test_feature: str, n_bins: int = 5
) -> go.Figure:
    """Bin the numeric feature into n_bins and plot the actual hit ratio and model predictions."""
    analysis2 = analysis.copy()
    analysis2.add_feature(test_feature)
    analysis2.fit()

    df = pd.concat([analysis.y, analysis.cv, analysis.X], axis=1)
    df = df.loc[df["fold"] >= 10]
    df[test_feature] = data.df[test_feature]
    df["current_model"] = analysis.yhat_proba(
        df.drop(columns=[analysis.y.name, analysis.cv.name, test_feature])
    )
    df["test_analysis"] = analysis2.yhat_proba(
        df.drop(columns=[analysis.y.name, analysis.cv.name])
    )

    bins = pd.qcut(df[test_feature], n_bins, duplicates="drop")
    df["bin"] = bins
    df["bin"] = df["bin"].astype(str)

    count_by_level = (
        df.groupby("bin")
        .count()[["hit_count"]]
        .reset_index()
        .rename(columns={"hit_count": "count"})
    )

    ave_by_level = (
        df.groupby("bin")
        .mean()[["hit_count", "current_analysis", "test_analysis"]]
        .reset_index()
        .set_index("bin")
        .join(count_by_level.set_index("bin"), on="bin")
        .reset_index()
        .sort_values("bin", ascending=True)
    )

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Bar(
            x=ave_by_level["bin"],
            y=ave_by_level["count"],
            name="Count",
            marker=get_marker_props("grey", 0.5, 0.5),
            showlegend=False,
        ),
        secondary_y=True,
    )

    fig.add_trace(
        go.Scattergl(
            x=ave_by_level["bin"],
            y=ave_by_level["hit_count"],
            mode="lines+markers",
            name="Actual",
            marker=get_marker_props(
                "blue", size=10, opacity=0.5, line=get_line_props("black", width=0.5)
            ),
            line=get_line_props("blue", width=2),
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scattergl(
            x=ave_by_level["bin"],
            y=ave_by_level["current_analysis"],
            mode="lines+markers",
            name="Current Analysis",
            marker=get_marker_props(
                "red", size=10, opacity=0.5, line=get_line_props("black", width=0.5)
            ),
            line=get_line_props("red", width=1.5, dash="dashdot"),
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scattergl(
            x=ave_by_level["bin"],
            y=ave_by_level["test_analysis"],
            mode="lines+markers",
            name="Test Analysis",
            marker=get_marker_props(
                "green", size=10, opacity=0.5, line=get_line_props("black", width=0.5)
            ),
            line=get_line_props("green", width=1.5, dash="dashdot"),
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
