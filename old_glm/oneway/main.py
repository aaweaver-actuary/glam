from typing import Protocol
from hit_ratio.old_glm.glm_data import BaseGlmData
from hit_ratio.old_glm.glm_result import BaseGlmResult
import pandas as pd
import plotly.graph_objects as go


class BaseResidualCalculator(Protocol):
    """Define an interface for calculating residuals."""

    def deviance_residuals(
        self, y: pd.Series, yhat_proba: pd.Series, groupby: pd.Series | None = None
    ) -> pd.DataFrame | pd.Series: ...

    def pearson_residuals(
        self, y: pd.Series, yhat_proba: pd.Series, groupby: pd.Series | None = None
    ) -> pd.DataFrame | pd.Series: ...

    def standardised_residuals(
        self, y: pd.Series, yhat_proba: pd.Series, groupby: pd.Series | None = None
    ) -> pd.DataFrame | pd.Series: ...

    def studentised_residuals(
        self, y: pd.Series, yhat_proba: pd.Series, groupby: pd.Series | None = None
    ) -> pd.DataFrame | pd.Series: ...

    def leverage(
        self, y: pd.Series, yhat_proba: pd.Series, groupby: pd.Series | None = None
    ) -> pd.DataFrame | pd.Series: ...

    def cooks_distance(
        self, y: pd.Series, yhat_proba: pd.Series, groupby: pd.Series | None = None
    ) -> pd.DataFrame | pd.Series: ...

    def influence(
        self, y: pd.Series, yhat_proba: pd.Series, groupby: pd.Series | None = None
    ) -> pd.DataFrame | pd.Series: ...


class BaseResidualPlotter(Protocol):
    """Define an interface for plotting residuals."""

    def plot_deviance_residuals(
        self, calculator: BaseResidualCalculator
    ) -> go.Figure: ...

    def plot_residuals_vs_fitted(
        self, calculator: BaseResidualCalculator
    ) -> go.Figure: ...

    def plot_qq(self, calculator: BaseResidualCalculator) -> go.Figure: ...

    def plot_scale_location(self, calculator: BaseResidualCalculator) -> go.Figure: ...

    def plot_leverage(self, calculator: BaseResidualCalculator) -> go.Figure: ...

    def plot_cooks_distance(self, calculator: BaseResidualCalculator) -> go.Figure: ...

    def plot_influence(self, calculator: BaseResidualCalculator) -> go.Figure: ...


class BaseOneWayCalculator(Protocol):
    """Define an interface for calculating one-way ANOVA."""

    def __init__(self, residuals: pd.Series, groupby: pd.Series) -> None: ...

    @property
    def residuals(self) -> pd.Series: ...

    @property
    def groupby(self) -> pd.Series: ...

    @property
    def groupby_levels(self) -> list[str]: ...

    @property
    def f_statistic(self) -> float: ...

    @property
    def p_value(self) -> float: ...


class BaseOneWayPlotter(Protocol):
    """Define an interface for plotting one-way ANOVA."""

    def plot(self) -> go.Figure: ...


class BaseOneWayRunner(Protocol):
    """Define an interface for running one-way ANOVA analysis."""

    def __init__(
        self,
        data: BaseGlmData,
        residual_calculator: BaseResidualCalculator,
        residual_plotter: BaseResidualPlotter,
        oneway_calculator: BaseOneWayCalculator,
        oneway_plotter: BaseOneWayPlotter,
    ) -> None: ...

    def run_analysis(
        self, data: pd.DataFrame, model: BaseGlmResult, groupby: str
    ) -> go.Figure: ...
