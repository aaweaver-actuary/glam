import pandas as pd

from glam.src.calculators.loglikelihood_calculators.binomial_loglikelihood_calculator import (
    BinomialLogLikelihoodCalculator,
)
from scipy.stats import chi2

__all__ = ["BinomialGlmLikelihoodRatioCalculator"]


class BinomialGlmLikelihoodRatioCalculator:
    """Concrete implementation of the BaseLikelihoodRatioCalculator for Binomial GLMs."""

    def __init__(
        self,
        y: pd.Series,
        yhat_proba: pd.Series,
        yhat_proba_new: pd.Series,
        k: int,
        k_new: int,
    ):
        self._y = y
        self._yhat_proba = yhat_proba
        self._yhat_proba_new = yhat_proba_new
        self._loglikelihood_calculator = BinomialLogLikelihoodCalculator(
            self._y, self._yhat_proba
        )
        self._loglikelihood_calculator_new = BinomialLogLikelihoodCalculator(
            self._y, self._yhat_proba_new
        )
        self._k = k
        self._k_new = k_new

    @property
    def y(self) -> pd.Series:
        return self._y

    @property
    def yhat_proba(self) -> pd.Series:
        return self._yhat_proba

    @property
    def yhat_proba_new(self) -> pd.Series:
        return self._yhat_proba_new

    @property
    def loglikelihood(self) -> pd.Series:
        return self._loglikelihood_calculator.calculate()

    @property
    def loglikelihood_new(self) -> pd.Series:
        return self._loglikelihood_calculator_new.calculate()

    @property
    def test_statistic(self) -> float:
        """Calculate the likelihood ratio test statistic."""
        return 2 * (self.loglikelihood_new.sum() - self.loglikelihood.sum())

    @property
    def degrees_of_freedom(self) -> int:
        """Calculate the degrees of freedom of the likelihood ratio test."""
        return self._k_new - self._k

    @property
    def p_value(self) -> float:
        """Calculate the p-value of the likelihood ratio test."""
        return 1 - chi2.cdf(self.test_statistic, self.degrees_of_freedom)

    def calculate(self) -> float:
        """Calculate the likelihood ratio test statistic."""
        return self.test_statistic

    def test(self) -> tuple:
        """Return the likelihood ratio test statistic and p-value."""
        return self.test_statistic, self.p_value
