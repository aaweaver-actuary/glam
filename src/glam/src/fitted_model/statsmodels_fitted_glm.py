import pandas as pd
from statsmodels.genmod.generalized_linear_model import (
    GLMResults as StatsmodelsGLMResults,
)

from glam.src.data.base_model_data import BaseModelData
from glam.src.enums import ModelType
from glam.src.fitted_model.base_fitted_model import BaseFittedModel


class StatsmodelsFittedGlm:
    """Concrete implementation of the GLM result functionality using the statsmodels library.

    Attributes
    ----------
    is_fitted : bool
        Whether the model has been fitted.
    data : BaseModelData
        The ModelData object containing the data used to fit the model.
    model : StatsmodelsGLMResults
        The fitted model object.
    model_type : ModelType
        The type of the model. See the `ModelType` enum for possible values. This
        attribute cannot be changed from the public API.
    coefficients : dict[str, float]
        The coefficients of the model besides the intercept (if present) in a dictionary.
    features : list[str]
        The features used to fit the model.
    intercept : float
        The intercept of the model.
    mu : pd.Series
        The expected value of the response variable. For a binary classification model,
        this is the probability of the positive class.
    residuals : pd.Series
        The residuals of the model.

    Methods
    -------
    **__init__(data: BaseModelData, model: BaseFittedModel | None) -> None**

        Initialize the object with the given data and model.
    **__repr__() -> str**

        Return a string representation of the object.
    **__str__() -> str**

        Return a string representation of the object.
    **yhat(X: pd.DataFrame | None = None) -> pd.Series**

        Return the predicted response variable. For a binary classification model, this is the predicted class.
    **yhat_proba(X: pd.DataFrame | None = None) -> pd.Series**

        Return the predicted response variable. For a binary classification model, this is the probability of the positive class.
    """

    def __init__(self, data: BaseModelData, model: BaseFittedModel | None) -> None:
        self._data = data
        self._model = model if model is not None else None
        self._model_type = ModelType.GLM

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model_type={self.model_type}, fitted={self.is_fitted})"

    def __str__(self) -> str:
        return self.__repr__()

    @property
    def is_fitted(self) -> bool:
        """Return whether the model has been fitted."""
        return self.model is not None

    @property
    def data(self) -> BaseModelData:
        """Return the ModelData object containing the data used to fit the model."""
        return self._data

    @property
    def model(self) -> StatsmodelsGLMResults:
        """Return the fitted model object."""
        return self._model

    @property
    def model_type(self) -> ModelType:
        """Return the type of the model.

        See the `ModelType` enum for possible values."""
        return self._model_type

    @property
    def coefficients(self) -> dict[str, float]:
        """Return the coefficients of the model besides the intercept (if present) in a dictionary."""

        params = self.model.params[self.model.params.index.to_series().ne("Intercept")]
        return dict(zip(params.index, params))

    @property
    def features(self) -> list[str]:
        """Return the features used to fit the model."""
        return list(self.coefficients.keys())

    @property
    def intercept(self) -> float:
        """Return the intercept of the model."""
        return self.model.params[0]

    @property
    def mu(self) -> pd.Series:
        """Return the expected value of the response variable.

        For a binary classification model, this is the probability of the positive class."""
        return pd.Series(self.model.mu, name="mu")

    @property
    def residuals(self) -> pd.Series:
        """Return the residuals of the model."""
        return pd.Series(self.model.resid_response)

    def yhat(self, X: pd.DataFrame | None = None) -> pd.Series:
        """Return the predicted response variable.

        For a binary classification model, this is the predicted class.

        Parameters
        ----------
        X : pd.DataFrame, optional
            The data to predict on. If not provided, the training data will be used.

        Returns
        -------
        pd.Series
            The predicted response variable.

        Examples
        --------
        >>> fitted_model = StatsmodelsFittedGlm(data, model)
        >>> fitted_model.yhat()
        0    1
        1    0
        2    1
        3    0
        4    1
        """
        if X is None:
            X = self.data.X
        return pd.Series(self.model.predict(X), name="yhat").round(0)

    def yhat_proba(self, X: pd.DataFrame | None = None) -> pd.Series:
        """Return the predicted response variable.

        For a binary classification model, this is the probability of the positive class.

        Parameters
        ----------
        X : pd.DataFrame, optional
            The data to predict on. If not provided, the training data will be used.

        Returns
        -------
        pd.Series
            The predicted probability of the positive class.

        Examples
        --------
        >>> fitted_model = StatsmodelsFittedGlm(data, model)
        >>> fitted_model.yhat_proba()
        0    0.8
        1    0.2
        2    0.9
        3    0.1
        4    0.7
        """
        if X is None:
            X = self.data.X
        return pd.Series(self.model.predict(X), name="yhat_proba")
