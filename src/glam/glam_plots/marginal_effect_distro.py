"""Plot the distribution of marginal effects of a feature or all features in a GLAM model."""

from __future__ import annotations
import plotly.graph_objects as go  # type: ignore

import polars as pl


def marginal_effect_distro(
    df: pl.LazyFrame, feature_name: str | None = None, by: str | None = None
) -> go.Figure:
    """Plot the distribution of marginal effects of a feature or all features in a GLAM model."""
    fig = go.Figure()

    if feature_name is not None:
        df = df.filter(pl.col("term") == feature_name)

    cols_to_drop_for_customdata = [
        "rowid",
        "contrast",
        "std_error",
        "statistic",
        "p_value",
        "conf_low",
        "s_value",
        "conf_high",
        "predicted",
        "predicted_lo",
        "predicted_hi",
    ]

    customdata = df.drop(cols_to_drop_for_customdata).collect().to_pandas()

    for col in customdata.columns:
        if col.startswith(("is_", "has_")):
            customdata[col] = customdata[col].round(0).astype(int)

    hovertemplate = "<b>dy/d[%{customdata[0]}] = %{customdata[1]:.1%}</b><br>"
    for i, dterm in enumerate(customdata.columns.tolist()[2:]):
        hovertemplate += f"<br><b>{dterm}:</b> "
        hovertemplate += "%{customdata["
        hovertemplate += f"{i+2}"
        if customdata.iloc[:, i].dtype == float:
            hovertemplate += "]:.2f}"
        else:
            hovertemplate += "]}"

    params = {
        "x": df.select("term").collect()["term"],
        "y": df.select("estimate").collect()["estimate"],
        "line": {"width": 1},
        "marker": {"opacity": 0.3, "line": {"width": 0}, "color": "firebrick"},
        "customdata": customdata,
        "hovertemplate": hovertemplate,
    }

    if by is None:
        fig.add_trace(
            go.Violin(**params) if feature_name is not None else go.Box(**params)
        )
    else:
        for level in df.select(pl.col(by)).unique().collect()[by]:
            params = {
                "x": df.filter(pl.col(by) == level).select(by).collect()[by],
                "y": df.filter(pl.col(by) == level)
                .select("estimate")
                .collect()["estimate"],
                "line": {"width": 1},
                "marker": {"opacity": 0.3, "line": {"width": 0}, "color": "firebrick"},
                "customdata": customdata.loc[customdata[by] == level],
                "hovertemplate": hovertemplate,
                "name": level,
                "legendgroup": f"Level of {by}",
            }

            fig.add_trace(
                go.Violin(**params) if feature_name is not None else go.Box(**params)
            )

    if feature_name is not None:
        layout = {
            "title": {"text": f"Distribution of marginal effects of [{feature_name}]"},
            "yaxis": {
                "title": f"<b>Hit ratio impact</b><br>Unit increase to <i>{feature_name}</i>"
            },
        }
    else:
        layout = {
            "title": {"text": "Distribution of marginal effects of all features"},
            "yaxis": {
                "title": "<b>Hit ratio impact</b><br>Unit increase to <i>each feature</i>"
            },
        }

    layout["width"] = 800
    layout["height"] = 800
    fig.update_layout(**layout)
    return fig
