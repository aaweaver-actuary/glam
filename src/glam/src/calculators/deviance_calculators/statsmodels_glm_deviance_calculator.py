"""Concrete implementation of the BaseDevianceCalculator for GLMs using the statsmodels library."""

import pandas as pd
import statsmodels.api as sm

__all__ = ["StatsmodelsGlmDevianceCalculator"]


class StatsmodelsGlmDevianceCalculator:
    """Concrete implementation of the BaseDevianceCalculator for GLMs using the statsmodels library."""

    def __init__(self, fitted_model: sm.GLM):
        self._fitted_model = fitted_model

    @property
    def fitted_model(self) -> pd.Series:
        return self._fitted_model

    def calculate(self) -> float:
        """Calculate the deviance of the GLM."""
        return self.fitted_model.deviance
