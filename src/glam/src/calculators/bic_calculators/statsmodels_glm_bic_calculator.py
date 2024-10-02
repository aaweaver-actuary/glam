"""Concrete implementation of the BaseBicCalculator for GLMs using the statsmodels library."""

import pandas as pd
import statsmodels.api as sm  # type: ignore

__all__ = ["StatsmodelsGlmBicCalculator"]


class StatsmodelsGlmBicCalculator:
    """Concrete implementation of the BaseBicCalculator for GLMs using the statsmodels library."""

    def __init__(self, fitted_model: sm.GLM):
        self._fitted_model = fitted_model

    @property
    def fitted_model(self) -> pd.Series:
        """Return the fitted model object."""
        return self._fitted_model

    def calculate(self) -> float:
        """Calculate the BIC of the GLM."""
        return self.fitted_model.models.model.bic_llf
