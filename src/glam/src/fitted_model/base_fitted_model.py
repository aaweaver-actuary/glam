"""Define an interface for the fitted model functionality we need, no matter the underlying library."""

from __future__ import annotations
from typing import Protocol
import pandas as pd
from glam.src.data.base_model_data import BaseModelData


class BaseFittedModel(Protocol):
    """The `BaseFittedModel` protocol defines the interface for the data structure needed to hold the fitted model object.

    Attributes
    ----------
    is_fitted : bool
        Whether the model has been fitted.
    data : BaseModelData
        The data used to fit the model.
    model : object
        The fitted model object.
    coefficients : dict[str, float]
        The coefficients of the model besides the intercept (if present).
    intercept : float
        The intercept of the model.
    mu : pd.Series
        The expected value of the response variable. For a binary classification model, this is the probability of the positive class.

    Methods
    -------
    **yhat(X: pd.DataFrame | None = None) -> pd.Series**

        Return the predicted response variable. For a binary classification model, this is the predicted class. If `X` is provided, return the predicted response variable for the given feature matrix. Otherwise, return the predicted response variable for the data used to fit the model.

    **yhat_proba(X: pd.DataFrame | None = None) -> pd.Series**

        Return the expected value of the response variable. For a binary classification model, this is the probability of the positive class. If `X` is provided, return the expected value of the response variable for the given feature matrix. Otherwise, return the expected value of the response variable for the data used to fit the model.
    """

    def __copy__(self):
        """Implement a deepcopy for a BaseFittedModel."""
        ...

    @property
    def is_fitted(self) -> bool:
        """Return whether the model has been fitted."""
        ...

    @property
    def data(self) -> BaseModelData:
        """Return the data used to fit the model."""
        ...

    @property
    def model(self) -> object:
        """Return the fitted model object."""
        ...

    @property
    def coefficients(self) -> dict[str, float]:
        """Return the coefficients of the model besides the intercept (if present)."""
        ...

    @property
    def intercept(self) -> float:
        """Return the intercept of the model."""
        ...

    @property
    def mu(self) -> pd.Series:
        """Return the expected value of the response variable."""
        ...

    def yhat(self, X: pd.DataFrame | None = None) -> pd.Series:
        """Return the predicted response variable.

        For a binary classification model, this is the predicted class. If `X` is provided, return the predicted response variable for the given feature matrix. Otherwise, return the predicted response variable for the data used to fit the model.
        """
        ...

    def yhat_proba(self, X: pd.DataFrame | None = None) -> pd.Series:
        """Return the expected value of the response variable.

        For a binary classification model, this is the probability of the positive class. If `X` is provided, return the expected value of the response variable for the given feature matrix. Otherwise, return the expected value of the response variable for the data used to fit the model.
        """
        ...
