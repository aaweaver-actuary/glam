"""Concrete implementation of the BaseAicCalculator for Binomial GLMs."""

import pandas as pd

from glam.src.calculators.loglikelihood_calculators.binomial_loglikelihood_calculator import (
    BinomialLogLikelihoodCalculator,
)

__all__ = ["BinomialGlmAicCalculator"]


class BinomialGlmAicCalculator:
    """Concrete implementation of the BaseAicCalculator for Binomial GLMs."""

    def __init__(self, y: pd.Series, yhat_proba: pd.Series, k: int):
        self._y = y
        self._yhat_proba = yhat_proba
        self._k = k
        self._loglikelihood_calculator = BinomialLogLikelihoodCalculator(
            self._y, self._yhat_proba
        )

    @property
    def y(self) -> pd.Series:
        """Return the response variable."""
        return self._y

    @property
    def yhat_proba(self) -> pd.Series:
        """Return the predicted probability of the positive class."""
        return self._yhat_proba

    @property
    def k(self) -> int:
        """Return the number of parameters in the model."""
        return self._k

    @property
    def loglikelihood(self) -> pd.Series:
        """Return the loglikelihood of the model."""
        return self._loglikelihood_calculator.calculate()

    def calculate(self) -> float:
        """Calculate the AIC of the Binomial GLM."""
        return 2 * self.k - 2 * self.loglikelihood.sum()
