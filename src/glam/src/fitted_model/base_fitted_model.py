"""This module provides the interface for the fitted model functionality we need, no matter the underlying library."""

from typing import Protocol
import pandas as pd


class BaseFittedModel(Protocol):
    """Protocol for the data structure that holds the results of a fitted model."""

    @property
    def coefficients(self) -> dict:
        """Return a dictionary with the coefficients of the model."""

    @property
    def intercept(self) -> float: ...

    @property
    def mu(self) -> pd.Series: ...

    def yhat(self, X: pd.DataFrame) -> pd.Series: ...

    def yhat_prob(self, X: pd.DataFrame) -> pd.Series: ...
