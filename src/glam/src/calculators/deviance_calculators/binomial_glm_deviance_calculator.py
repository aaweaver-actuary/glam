"""Concrete implementation of the BaseDevianceCalculator for Binomial GLMs."""

import pandas as pd

from glam.src.calculators.loglikelihood_calculators.binomial_loglikelihood_calculator import (
    BinomialLogLikelihoodCalculator,
)


class BinomialGlmDevianceCalculator:
    """Concrete implementation of the BaseDevianceCalculator for Binomial GLMs."""

    def __init__(self, y: pd.Series, yhat_proba: pd.Series):
        self._y = y
        self._yhat_proba = yhat_proba
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
    def loglikelihood(self) -> pd.Series:
        """Return the loglikelihood of the model."""
        return self._loglikelihood_calculator.calculate()

    def calculate(self) -> float:
        """Calculate the deviance of the Binomial GLM."""
        return 2 * self.loglikelihood.sum()
