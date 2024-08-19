import statsmodels
import statsmodels.formula.api as smf

from glam.src.data.base_model_data import BaseModelData
from glam.src.fitted_model.base_fitted_model import BaseFittedModel
from glam.src.fitted_model.statsmodels_fitted_glm import StatsmodelsFittedGlm
from glam.src.enums import ModelTask


class StatsmodelsFormulaGlmFitter:
    """Implement the BaseModelFitter protocol to fit a GLM model using the statsmodels library with a formula."""

    def __init__(
        self,
        data: BaseModelData,
        fitted_model: BaseFittedModel = StatsmodelsFittedGlm,
        task: ModelTask = ModelTask.CLASSIFICATION,
    ):
        self._data = data
        self._fitted_model = fitted_model(data, None)
        self._task = task

    @property
    def data(self):
        return self._data

    @property
    def df(self):
        return self.data.df

    @property
    def fitted_model(self):
        return self._fitted_model

    @fitted_model.setter
    def fitted_model(self, fitted_model: BaseFittedModel):
        self._fitted_model = fitted_model

    @property
    def task(self):
        return self._task

    def _fit_classifier(self, formula: str) -> BaseFittedModel:
        from statsmodels.genmod.families.family import Binomial

        model = smf.glm(formula=formula, data=self.df, family=Binomial())
        return model.fit()

    def _fit_regressor(self, formula: str) -> BaseFittedModel:
        from statsmodels.genmod.families.family import Gamma

        model = smf.glm(formula=formula, data=self.df, family=Gamma())
        return model.fit()

    def fit(
        self,
        formula: str,
    ) -> statsmodels.genmod.generalized_linear_model.GLM:
        fitted_model = (
            self._fit_classifier(formula)
            if self.task == ModelTask.CLASSIFICATION
            else self._fit_regressor(formula)
        )

        self.fitted_model = self._fitted_model.__class__(self.data, fitted_model)
