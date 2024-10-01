"""Concrete implementation of the BaseDevianceResidualCalculator for Binomial GLMs."""

from __future__ import annotations
import pandas as pd
import numpy as np

from glam.src.calculators.loglikelihood_calculators.binomial_loglikelihood_calculator import (
    BinomialLogLikelihoodCalculator,
)
from glam.src.calculators.hat_matrix_calculators.binomial_glm_hat_matrix_calculator import (
    BinomialGlmHatMatrixCalculator,
)
from glam.src.calculators.leverage_calculators.binomial_glm_leverage_calculator import (
    BinomialGlmLeverageCalculator,
)


class BinomialGlmResidualCalculator:
    """Concrete implementation of the BaseDevianceResidualCalculator for Binomial GLMs."""

    def __init__(
        self,
        X: pd.DataFrame | None = None,
        y: pd.Series | None = None,
        yhat_proba: pd.Series | None = None,
        beta: pd.Series | None = None,
    ):
        self._X = X
        self._y = y
        self._yhat_proba = yhat_proba
        self._beta = beta

    @property
    def X(self) -> pd.DataFrame:
        """Return the design matrix."""
        if self._X is None:
            raise ValueError(
                "X is not set in the constructor of the BinomialGlmResidualCalculator."
            )

        return self._X

    @property
    def y(self) -> pd.Series:
        """Return the response variable."""
        if self._y is None:
            raise ValueError(
                "y is not set in the constructor of the BinomialGlmResidualCalculator."
            )

        return self._y

    @property
    def yhat_proba(self) -> pd.Series:
        """Return the predicted probabilities."""
        if self._yhat_proba is None:
            raise ValueError(
                "yhat_proba is not set in the constructor of the BinomialGlmResidualCalculator."
            )

        return pd.Series(self._yhat_proba, index=self.y.index, name="yhat_proba")

    @property
    def beta(self) -> pd.Series:
        """Return the estimated coefficients."""
        if self._beta is None:
            raise ValueError(
                "beta is not set in the constructor of the BinomialGlmResidualCalculator."
            )

        return pd.Series(self._beta, index=self.X.columns, name="beta")

    @property
    def hat_matrix(self) -> np.ndarray:
        """Return the hat matrix."""
        """Calculate the hat matrix."""
        calculator = BinomialGlmHatMatrixCalculator(self.X, self.yhat_proba)
        return calculator.weight_matrix

    @property
    def loglikelihood(self) -> pd.Series:
        """Calculate the loglikelihood."""
        calculator = BinomialLogLikelihoodCalculator(self.y, self.yhat_proba)
        return calculator.calculate()

    @property
    def saturated_loglikelihood(self) -> pd.Series:
        """Calculate the saturated loglikelihood."""
        calculator = BinomialLogLikelihoodCalculator(self.y, self.y)
        return calculator.calculate()

    @property
    def unit_deviance(self) -> pd.Series:
        """Calculate the unit deviance."""
        return pd.Series(
            2 * (self.saturated_loglikelihood - self.loglikelihood),
            name="Unit Deviance",
        )

    @property
    def sign_function(self) -> pd.Series:
        """Return the sign of the response variable."""
        return pd.Series(np.where(self.y.eq(1), 1, -1), name="Sign Function")

    @property
    def variance_function(self) -> pd.Series:
        """Defines the variance to mean relationship for the binomial distribution."""
        return pd.Series(
            (self.yhat_proba * (1 - self.yhat_proba)).to_numpy(), name="V(mu)"
        )

    @property
    def leverage_calculator(self) -> BinomialGlmLeverageCalculator:
        """Return the leverage calculator."""
        return BinomialGlmLeverageCalculator(self.X, self.yhat_proba)

    def deviance_residuals(self) -> pd.Series:
        """Calculate the deviance residuals."""
        return pd.Series(
            2 * self.sign_function * np.sqrt(self.unit_deviance),
            index=self.y.index,
            name="Deviance Residuals",
        )

    def pearson_residuals(self, std: bool = True) -> pd.Series:
        """Calculate the Pearson residuals."""
        unstandardized = (self.y - self.yhat_proba) / np.sqrt(self.variance_function)

        if std:
            return pd.Series(
                unstandardized / np.sqrt(1 - self.leverage_calculator.calculate()),
                name="Pearson Residuals (Standardized)",
            )

        return pd.Series(unstandardized, name="Pearson Residuals")

    def partial_residuals(self, feature: str) -> pd.Series:
        """Return the partial residuals for a given feature."""
        residuals = self.deviance_residuals()

        feature_index = self.X.columns.get_loc(feature)
        X_j = self.X.iloc[:, feature_index]
        beta_j = self.beta.iloc[feature_index]

        return pd.Series(
            residuals - X_j * beta_j,
            index=self.X.index,
            name=f"Partial Residuals/{feature}",
        )
