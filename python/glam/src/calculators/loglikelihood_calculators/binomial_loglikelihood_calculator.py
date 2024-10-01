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
        return pd.Series(self._y, index=self._y.index, name="y")

    @property
    def yhat_proba(self) -> pd.Series:
        return (
            pd.Series(self._yhat_proba, index=self.y.index, name="yhat_proba")
            .replace(0, 1e-6)
            .replace(1, 1 - 1e-6)
        )

    def calculate(self) -> pd.Series:
        """Calculate the cross-entropy loss."""
        ln_p = pd.Series(np.log(self.yhat_proba), index=self.y.index)
        ln_1_p = pd.Series(np.log(1 - self.yhat_proba), index=self.y.index)
        return pd.Series(
            self.y * ln_p + (1 - self.y) * ln_1_p,
            index=self.y.index,
            name="loglikelihood",
        )
