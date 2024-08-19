import pandas as pd
import plotly.graph_objects as go
from scipy.stats import norm
from abc import ABC, abstractmethod
from glam.src.data.base_model_data import BaseModelData
from glam.src.fitted_model.base_fitted_model import BaseFittedModel

__all__ = ["BaseDevianceResid"]


class BaseDevianceResid(ABC):
    """Provides a base class & functionality for calculating deviance residuals. Subclasses should implement the `log_likelihood` method."""

    def __init__(
        self, data: BaseModelData, fitted_model: BaseFittedModel, features: list[str]
    ):
        self._data = data
        self._fitted_model = fitted_model
        self._features = features

    @property
    def features(self):
        return self._features

    @property
    def X(self):
        return self._data.X[self.features]

    @property
    def y(self):
        return self._data.y

    def yhat(self, X: pd.DataFrame | None = None):
        if X is None:
            X = self.X
        return self._fitted_model.yhat(X)

    def yhat_proba(self, X: pd.DataFrame | None = None):
        if X is None:
            X = self.X
        return self._fitted_model.yhat_proba(X)

    @abstractmethod
    def log_likelihood(self, y: pd.Series, yhat_proba: pd.Series) -> pd.Series:
        """Return the log-likelihood of the model."""
        pass

    def loglik_saturated_model(self, y: pd.Series) -> pd.Series:
        """Return the log-likelihood of the saturated model."""
        return self.log_likelihood(y, y)

    def loglik_null_model(self, y: pd.Series) -> pd.Series:
        """Return the log-likelihood of the null model."""
        return self.log_likelihood(y, y.mean())

    def deviance_residuals(self, groupby: pd.Series | None = None) -> pd.Series:
        """Return the deviance residuals."""
        yhat_proba = self.yhat_proba()
        loglik = self.log_likelihood(self.y, yhat_proba)
        loglik_saturated = self.loglik_saturated_model(self.y)
        return -2 * (loglik - loglik_saturated)

    def qq_plot(self, color_by: str | None = None) -> go.Figure:
        """Return a QQ plot of the deviance residuals."""
        deviance_residuals = self.deviance_residuals()
        deviance_residuals = deviance_residuals.sort_values()
        n = len(deviance_residuals)
        quantiles = norm.ppf((1 + pd.Series(range(n))) / (n + 1))
        fig = go.Figure()

        fig.add_trace(
            go.Scattergl(
                x=quantiles,
                y=deviance_residuals,
                mode="markers",
                marker=dict(
                    size=5,
                    color=color_by,
                ),
            ),
            layout=dict(
                title="QQ Plot",
                xaxis=dict(title="Theoretical Quantiles"),
                yaxis=dict(title="Deviance Residuals"),
            ),
        )
