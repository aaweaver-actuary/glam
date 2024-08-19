import statsmodels
import statsmodels.formula.api as smf
import pandas as pd

from glam.src.fitted_model.base_fitted_model import BaseFittedModel
from glam.src.enums import ModelTask


class StatsmodelsFormulaGlmFitter:
    """Implement the BaseModelFitter protocol to fit a GLM model using the statsmodels library with a formula.

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
        fitted_model: BaseFittedModel | None = None,
        task: ModelTask = ModelTask.CLASSIFICATION,
    ):
        self._fitted_model = fitted_model
        self._task = task

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(task={self.task})"

    def __str__(self) -> str:
        return self.__repr__()

    @property
    def fitted_model(self):
        return self._fitted_model

    @fitted_model.setter
    def fitted_model(self, fitted_model: BaseFittedModel):
        self._fitted_model = fitted_model

    @property
    def task(self):
        return self._task

    def _fit_classifier(
        self, formula: str, X: pd.DataFrame, y: pd.Series
    ) -> BaseFittedModel:
        from statsmodels.genmod.families.family import Binomial

        model = smf.glm(
            formula=formula, data=pd.concat([y, X], axis=1), family=Binomial()
        )
        return model.fit()

    def _fit_regressor(
        self, formula: str, X: pd.DataFrame, y: pd.Series
    ) -> BaseFittedModel:
        from statsmodels.genmod.families.family import Gamma

        model = smf.glm(formula=formula, data=pd.concat([y, X], axis=1), family=Gamma())
        return model.fit()

    def fit(
        self,
        formula: str,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> statsmodels.genmod.generalized_linear_model.GLM:
        fitted_model = (
            self._fit_classifier(formula, X, y)
            if self.task == ModelTask.CLASSIFICATION
            else self._fit_regressor(formula, X, y)
        )

        self.fitted_model = fitted_model
        return fitted_model
