from typing import Protocol

import numpy as np
import statsmodels
from sklearn.metrics import roc_auc_score, roc_curve

from hit_ratio.old_glm.model_data import BaseModelData





class StatsmodelsGlmResult:
    """This class provides a concrete implementation of the GLM result functionality using the statsmodels library."""

    def __init__(
        self,
        data: BaseModelData,
    ):
        self.data = data

    def coefficients(
        self, model: statsmodels.genmod.generalized_linear_model.GLMResults
    ) -> np.ndarray:
        """Return the coefficients of the model besides the intercept (if present)."""
        return model.params[1:]

    def intercept(
        self, model: statsmodels.genmod.generalized_linear_model.GLMResults
    ) -> float:
        """Return the intercept of the model."""
        return model.params[0]

    def mu(
        self, model: statsmodels.genmod.generalized_linear_model.GLMResults
    ) -> np.ndarray:
        """Return the expected value of the response variable. For a binary classification model, this is the probability of the positive class."""
        return model.mu

    def y_pred(
        self, model: statsmodels.genmod.generalized_linear_model.GLMResults
    ) -> np.ndarray:
        """Return the predicted response variable. For a binary classification model, this is the predicted class."""
        return model.predict()

    def residuals(
        self, model: statsmodels.genmod.generalized_linear_model.GLMResults
    ) -> np.ndarray:
        """Return the residuals of the model."""
        return model.resid_response

    def yhat(
        self,
        X: np.ndarray,
        model: statsmodels.genmod.generalized_linear_model.GLMResults,
    ) -> np.ndarray:
        return model.predict(X)

    def yhat_prob(
        self,
        X: np.ndarray,
        model: statsmodels.genmod.generalized_linear_model.GLMResults,
    ) -> np.ndarray:
        return model.predict(X)
