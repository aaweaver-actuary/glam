"""Implement a concrete implementation of a fitter for a GLM model using the statsmodels library with the formula API.

This module implements the StatsmodelsFormulaGlmFitter class, which is a concrete implementation of a fitter for a GLM model using the statsmodels library with the formula API. It provides a fit method to fit a GLM model using a formula and data.
"""

from __future__ import annotations
import statsmodels  # type: ignore
import statsmodels.formula.api as smf  # type: ignore
import pandas as pd

from glam.src.fitted_model.statsmodels_fitted_glm import StatsmodelsFittedGlm
from glam.src.enums import ModelTask


class StatsmodelsFormulaGlmFitter:
    """Implement a fitter for a GLM model using the statsmodels library with the formula API.

    This is the standard way to fit a GLM model in glam. It uses the statsmodels library to fit a GLM model using a formula.

    Attributes
    ----------
    fitted_model : BaseFittedModel | None
        The fitted model. If None, the model has not been fitted yet.
    task : ModelTask
        The task of the model (classification or regression). Uses the ModelTask enum.

    Methods
    -------
    **fit(formula: str, X: pd.DataFrame, y: pd.Series) -> statsmodels.genmod.generalized_linear_model.GLM**

        Fit a GLM model using the formula and data.
    """

    def __init__(
        self,
        fitted_model: StatsmodelsFittedGlm | None = None,
        task: ModelTask = ModelTask.CLASSIFICATION,
    ):
        self._fitted_model = fitted_model
        self._task = task

    def __repr__(self) -> str:  # noqa: D105
        return f"{self.__class__.__name__}(task={self.task})"

    def __str__(self) -> str:  # noqa: D105
        return self.__repr__()

    @property
    def fitted_model(self) -> StatsmodelsFittedGlm | None:
        """Return the fitted model, or None if the model has not been fitted yet."""
        return self._fitted_model

    @fitted_model.setter
    def fitted_model(self, fitted_model: StatsmodelsFittedGlm) -> None:
        """Set the fitted model."""
        self._fitted_model = fitted_model

    @property
    def task(self) -> ModelTask:
        """Return the task of the model (classification or regression)."""
        return self._task

    def _fit_classifier(
        self, formula: str, X: pd.DataFrame, y: pd.Series
    ) -> StatsmodelsFittedGlm:
        """Fit a classifier using the formula and data."""
        from statsmodels.genmod.families.family import Binomial  # type: ignore

        model = smf.glm(
            formula=formula, data=pd.concat([y, X], axis=1), family=Binomial()
        )
        return model.fit()

    def _fit_regressor(
        self, formula: str, X: pd.DataFrame, y: pd.Series
    ) -> StatsmodelsFittedGlm:
        """Fit a regressor using the formula and data."""
        from statsmodels.genmod.families.family import Gamma  # type: ignore

        model = smf.glm(formula=formula, data=pd.concat([y, X], axis=1), family=Gamma())
        return model.fit()

    def fit(
        self, formula: str, X: pd.DataFrame, y: pd.Series
    ) -> statsmodels.genmod.generalized_linear_model.GLM:
        """Fit a GLM model using the formula and data.

        Parameters
        ----------
        formula : str
            The formula to use to fit the model.
        X : pd.DataFrame
            The features to use to fit the model.
        y : pd.Series
            The target to use to fit the model.

        Returns
        -------
        statsmodels.genmod.generalized_linear_model.GLM
            The fitted model.
        """
        fitted_model = (
            self._fit_classifier(formula, X, y)
            if self.task == ModelTask.CLASSIFICATION
            else self._fit_regressor(formula, X, y)
        )

        self.fitted_model = fitted_model
        return fitted_model
