"""Define a concrete implementation of the BaseLikelihoodRatioCalculator for Binomial GLMs."""

from __future__ import annotations

import pandas as pd
import numpy as np

from glam.src.calculators.hat_matrix_calculators.binomial_glm_hat_matrix_calculator import (
    BinomialGlmHatMatrixCalculator,
)

__all__ = ["BinomialGlmLeverageCalculator"]


class BinomialGlmLeverageCalculator:
    """Define a concrete implementation of the BaseLikelihoodRatioCalculator for Binomial GLMs."""

    def __init__(self, X: pd.DataFrame, yhat_proba: pd.Series):
        self._X = X
        self._yhat_proba = yhat_proba

    @property
    def X(self) -> pd.DataFrame:
        """Return the feature matrix."""
        return self._X

    @property
    def yhat_proba(self) -> pd.Series:
        """Return the predicted probability of the positive class."""
        return self._yhat_proba

    @property
    def hat_matrix(self) -> np.ndarray:
        """Calculate the hat matrix."""
        calculator = BinomialGlmHatMatrixCalculator(self.X, self.yhat_proba)
        return calculator.calculate()

    def _leverage_single(self, index: int) -> float:
        """Calculate the leverage for a single observation."""
        return self.hat_matrix[index, index]

    def _leverage_all(self) -> pd.Series:
        """Calculate the leverage for all observations."""
        return pd.Series(np.diag(self.hat_matrix), index=self.X.index)

    def calculate(self, index: int | None = None) -> float | pd.Series:
        """Calculate the leverage for a single observation or all observations."""
        if index is None:
            return self._leverage_all()

        return self._leverage_single(index)

    def calculate_all(self) -> pd.Series:
        """Calculate the leverage for all observations."""
        return self._leverage_all()
