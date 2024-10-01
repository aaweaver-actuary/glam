"""Concrete implementation of the DevianceCalculator interface for the null model deviance."""

import pandas as pd
import numpy as np
from glam.src.calculators.loglikelihood_calculators.base_loglikelihood_calculator import (
    BaseLogLikelihoodCalculator,
)
from glam.src.calculators.loglikelihood_calculators.binomial_loglikelihood_calculator import (
    BinomialLogLikelihoodCalculator,
)


class NullModelDevianceCalculator:
    """Concrete implementation of the DevianceCalculator interface for the null model deviance."""

    def __init__(self, y: pd.Series, loglikelihood_calculator: BaseLogLikelihoodCalculator | None = None):
        self._y = y
        self._yhat_proba = pd.Series(np.full_like(y, y.mean())) # This is the null model -- the mean of the observed data
        self._loglikelihood_calculator = (
            loglikelihood_calculator
            if loglikelihood_calculator
            else BinomialLogLikelihoodCalculator(self._y, self._yhat_proba)
        )

    @property
    def y(self) -> pd.Series:
        return self._y
    
    @property
    def yhat_proba(self) -> pd.Series:
        return self._yhat_proba
    
    @property
    def loglikelihood(self) -> pd.Series:
        return self._loglikelihood_calculator.calculate()

    def calculate(self) -> float:
        """Calculate the null model deviance."""
        return 2 * self.loglikelihood.sum()