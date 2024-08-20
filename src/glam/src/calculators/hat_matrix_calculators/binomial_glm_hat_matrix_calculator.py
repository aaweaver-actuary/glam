import pandas as pd
import numpy as np
from glam.src.calculators.hat_matrix_calculators.base_hat_matrix_calculator import (
    BaseHatMatrixCalculator,
)


class BinomialGlmHatMatrixCalculator(BaseHatMatrixCalculator):
    def __init__(self, X: pd.DataFrame, yhat_proba: pd.Series):
        self._X = X
        self._yhat_proba = yhat_proba

    @property
    def X(self) -> np.ndarray:
        return self._X.to_numpy()

    @property
    def fitted_variance(self) -> np.ndarray:
        """Defines the variance to mean relationship for the binomial distribution."""
        phat = self._yhat_proba.to_numpy()
        return phat * (1 - phat)

    @property
    def weight_matrix(self) -> np.ndarray:
        """The weight matrix for the binomial GLM is a diagonal matrix with diagonal elements equal to the fitted variance."""
        return np.diag(self.fitted_variance)
