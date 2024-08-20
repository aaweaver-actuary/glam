"""Concrete implementation of the LogLikelihoodCalculator interface for the binomial likelihood."""

import pandas as pd
import numpy as np

__all__ = ["BinomialLogLikelihoodCalculator"]


class BinomialLogLikelihoodCalculator:
    """Concrete implementation of the LogLikelihoodCalculator interface for the binomial likelihood."""

    def __init__(self, y: pd.Series, yhat_proba: pd.Series):
        self._y = y
        self._yhat_proba = yhat_proba

    @property
    def y(self) -> pd.Series:
        return self._y

    @property
    def yhat_proba(self) -> pd.Series:
        return self._yhat_proba

    def calculate(self) -> pd.Series:
        """Calculate the cross-entropy loss."""
        return pd.Series(
            -self.y * np.log(self.yhat_proba)
            - (1 - self.y) * np.log(1 - self.yhat_proba),
        )
