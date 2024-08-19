from typing import Protocol
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats


class BaseResidualCalculator(Protocol):
    """Define a protocol interface for calculating different types of residuals, and for defining useful methods for plotting."""

    def calculate(self, y: pd.Series, yhat: pd.Series) -> pd.Series: ...

    def plot(self, y: pd.Series, yhat: pd.Series) -> None: ...

    def hist(self, y: pd.Series, yhat: pd.Series) -> None: ...


class DevianceResidualCalculator:
    """Calculate deviance residuals."""

    def calculate(self, y: pd.Series, yhat: pd.Series) -> pd.Series:
        """Calculate deviance residuals."""
        return 2 * (y * np.log(y / yhat) + (1 - y) * np.log((1 - y) / (1 - yhat)))

    def hist(self, y: pd.Series, yhat: pd.Series) -> None:
        """Plot a histogram of deviance residuals."""
        resid = self.calculate(y, yhat)
        mean_resid, std_resid = np.mean(resid), np.std(resid)

        kde = stats.gaussian_kde(resid)

        xaxis_values = np.linspace(min(resid), max(resid), 500)
        fitted_normal_density = stats.norm.pdf(xaxis_values, mean_resid, std_resid)
        fitted_kde = kde(xaxis_values)

        fig = go.Figure()

        fig.add_trace(
            go.Histogram(
                x=resid,
                histnorm="probability density",
                name="Histogram",
                marker=dict(
                    color="lightblue", line=dict(color="black", width=0.5), opacity=0.75
                ),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=xaxis_values,
                y=fitted_normal_density,
                mode="lines",
                name="Fitted Normal Density",
                line=dict(color="red", width=2),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=xaxis_values,
                y=fitted_kde,
                mode="lines",
                name="Fitted KDE",
                line=dict(color="green", width=2),
            )
        )

        fig.update_layout(title="Deviance Residuals Histogram")

    def plot(self, y: pd.Series, yhat: pd.Series) -> None:
        """Plot deviance residuals."""
        pass


class PearsonResidualCalculator:
    """Calculate Pearson residuals."""

    def calculate(self, y: pd.Series, yhat: pd.Series) -> pd.Series:
        """Calculate Pearson residuals."""
        return (y - yhat) / np.sqrt(yhat * (1 - yhat))


class StandardisedResidualCalculator:
    """Calculate standardised residuals."""

    def calculate(self, y: pd.Series, yhat: pd.Series) -> pd.Series:
        """Calculate standardised residuals."""
        return (y - yhat) / np.sqrt(yhat * (1 - yhat) * (1 - y.size))
