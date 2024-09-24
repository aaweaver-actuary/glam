"""Define a function to plot partial residual plots."""

from re import T
import plotly.graph_objects as go
import statsmodels.api as sm


def logistic_regression_partial_residual_plot(model: sm.GLM, feature: str) -> go.Figure:
    """Plot the partial residuals for a feature in a logistic regression model.

    Parameters
    ----------
    model : sm.GLM
        The logistic regression model.
    feature : str
        The feature to plot the partial residuals for.

    Returns
    -------
    go.Figure
        The plot of the partial residuals.
    """
    # Get the predicted probabilities
    predicted_probabilities = model.predict(model.exog)

    # Get the residuals
    residuals = model.endog - predicted_probabilities

    # Create the partial residuals
    partial_residuals = (
        residuals
        + model.exog[:, model.exog_names.index(feature)] * model.params[feature]
    )

    # Create the figure
    fig = go.Figure()

    # Add the scatter plot of the feature against the partial residuals
    fig.add_trace(
        go.Scatter(
            x=model.exog[:, model.exog_names.index(feature)],
            y=partial_residuals,
            mode="markers",
            name="Partial Residuals",
        )
    )

    # Add the line of best fit
    fig.add_trace(
        go.Scatter(
            x=model.exog[:, model.exog_names.index(feature)],
            y=model.exog[:, model.exog_names.index(feature)] * model.params[feature],
            mode="lines",
            name="Line of Best Fit",
        )
    )

    # Annotate the plot with the equation for the partial residuals
    equation = f"Partial Residual = Residual + {feature} * {model.params[feature]:.3f}"
    fig.add_annotation(
        x=0.05,
        y=0.95,
        xref="paper",
        yref="paper",
        text=equation,
        # Point to the line of best fit
        showarrow=True,
        arrowhead=1,
        ax=0,
        ay=-40,
    )

    # Set the title
    title = f"Partial Residual Plot for {feature}"

    # Add the axes labels
    fig.update_layout(xaxis_title=feature, yaxis_title="Partial Residuals", title=title)

    return fig
