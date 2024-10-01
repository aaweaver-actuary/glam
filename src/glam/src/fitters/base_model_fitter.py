"""Define a protocol interface for fitting a linear model model (GLM, GEE, GLMM, etc.)."""

import pandas as pd
from typing import Protocol
from glam.src.fitted_model.base_fitted_model import BaseFittedModel


class BaseModelFitter(Protocol):
    """Define a protocol interface for fitting a linear model model (GLM, GEE, GLMM, etc.).

    Methods
    -------
    **fit(X: pd.DataFrame, y: pd.Series) -> BaseFittedModel**

        Fit the model to the data.
    """

    def fit(self, X: pd.DataFrame, y: pd.Series) -> BaseFittedModel:
        """Fit the model to the data."""
        ...
