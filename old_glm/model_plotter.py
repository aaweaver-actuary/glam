from typing import Protocol
import plotly.graph_objects as go

from hit_ratio.old_glm.model_result import BaseModelResult
from hit_ratio.old_glm.model_data import BaseModelData
from hit_ratio.old_glm.residual_calculator import DevianceResidualCalculator

import pandas as pd


class BaseModelPlotter(Protocol):
    def plot(self) -> go.Figure: ...


class GlmPlotter:
    def __init__(self, data: BaseModelData, model: BaseModelResult) -> None:
        self._model = model
        self._data = data
        self._deviance_residual_calculator = DevianceResidualCalculator()

    @property
    def y(self) -> pd.Series:
        return self._data.y

    @property
    def yhat(self) -> pd.Series:
        return self._model.yhat(self._data.X)

    def deviance_residual_histogram(self) -> go.Figure:
        return self._deviance_residual_calculator.hist(self.y, self.yhat)

    def deviance_residual_plot(self) -> go.Figure:
        return self._deviance_residual_calculator.plot(self.y, self.yhat)

    def plot(self) -> go.Figure:
        return self.model.plot()
