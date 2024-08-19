import pandas as pd
from statsmodels.genmod.generalized_linear_model import (
    GLMResults as StatsmodelsGLMResults,
)

from glam.src.data.base_model_data import BaseModelData
from glam.src.enums import ModelType
from glam.src.fitted_model.base_fitted_model import BaseFittedModel


class StatsmodelsFittedGlm:
    """This class provides a concrete implementation of the GLM result functionality using the statsmodels library."""

    def __init__(self, data: BaseModelData, model: BaseFittedModel | None) -> None:
        self._data = data
        self._model = model if model is not None else Stats
        self._model_type = ModelType.GLM

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
        """Return the expected value of the response variable. For a binary classification model, this is the probability of the positive class."""
        return pd.Series(self.model.mu, name="mu")

    @property
    def residuals(self) -> pd.Series:
        """Return the residuals of the model."""
        return pd.Series(self.model.resid_response)

    def yhat(self, X: pd.DataFrame | None = None) -> pd.Series:
        if X is None:
            X = self.data.X
        return pd.Series(self.model.predict(X), name="yhat").round(0)

    def yhat_proba(self, X: pd.DataFrame | None = None) -> pd.Series:
        if X is None:
            X = self.data.X
        return pd.Series(self.model.predict(X), name="yhat_proba")
