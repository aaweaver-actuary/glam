"""This module provides the interface for the fitted model functionality we need, no matter the underlying library."""

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

    @property
    def is_fitted(self) -> bool: ...

    @property
    def data(self) -> BaseModelData: ...

    @property
    def model(self) -> object: ...

    @property
    def coefficients(self) -> dict[str, float]: ...

    @property
    def intercept(self) -> float: ...

    @property
    def mu(self) -> pd.Series: ...

    def yhat(self, X: pd.DataFrame | None = None) -> pd.Series: ...

    def yhat_proba(self, X: pd.DataFrame | None = None) -> pd.Series: ...
