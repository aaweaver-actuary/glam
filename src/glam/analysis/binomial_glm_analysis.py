import pandas as pd
import logging

from glam.analysis.base_analysis import BaseAnalysis
from glam.src.calculators.deviance_calculators.statsmodels_glm_deviance_calculator import (
    StatsmodelsGlmDevianceCalculator,
)
from glam.src.data.base_model_data import BaseModelData
from glam.src.enums.model_task import ModelTask
from glam.src.fitters.base_model_fitter import BaseModelFitter
from glam.src.model_list.base_model_list import BaseModelList
from glam.src.fitted_model.base_fitted_model import BaseFittedModel
from glam.src.data.data_prep import BaseDataSplitter, BasePreprocessor
from glam.src.fitters.statsmodels_formula_glm_fitter import StatsmodelsFormulaGlmFitter
from glam.src.model_list.default_model_list import DefaultModelList
from glam.src.data.data_prep import TimeSeriesDataSplitter, DefaultPreprocessor
from glam.src.calculators.residual_calculators.binomial_glm_residual_calculator import (
    BinomialGlmResidualCalculator,
)
from glam.src.calculators.aic_calculators.statsmodels_glm_aic_calculator import (
    StatsmodelsGlmAicCalculator,
)
from glam.src.calculators.leverage_calculators.binomial_glm_leverage_calculator import (
    BinomialGlmLeverageCalculator,
)
from glam.src.analysis_of_deviance.analysis_of_deviance_feature_evaluators.binomial_glm_analysis_of_deviance_feature_evaluator import (
    BinomialGlmAnalysisOfDevianceFeatureEvaluator,
)

__all__ = ["BinomialGlmAnalysis"]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BinomialGlmAnalysis(BaseAnalysis):
    def __init__(
        self,
        data: BaseModelData,
        fitter: BaseModelFitter | None = None,
        models: BaseModelList | None = None,
        features: list[str] | None = None,
        interactions: list[str] | None = None,
        fitted_model: BaseFittedModel | None = None,
        splitter: BaseDataSplitter | None = None,
        preprocessor: BasePreprocessor | None = None,
        task: ModelTask = ModelTask.CLASSIFICATION,
    ):
        super().__init__(
            data,
            fitter,
            models,
            features,
            interactions,
            fitted_model,
            splitter,
            preprocessor,
            task,
        )

        self._data = data
        self._fitter = fitter if fitter is not None else StatsmodelsFormulaGlmFitter()
        self._models = models if models is not None else DefaultModelList()
        self._fitted_model = fitted_model
        self._features = features if features is not None else []
        self._interactions = interactions if interactions is not None else []

        self._splitter = (
            splitter if splitter is not None else TimeSeriesDataSplitter(data)
        )
        self._preprocessor = (
            preprocessor if preprocessor is not None else DefaultPreprocessor(data)
        )

        self._task = task

    def __repr__(self):
        if len(self.features) > 0:
            return f"BinaryGlmAnalysis({self.linear_formula})"

        return f"BinaryGlmAnalysis({self.data.y.name} ~ 1)"

    @property
    def mu(self) -> pd.Series:
        """Return the expected value of the response variable."""
        return pd.Series(self.models.model.mu, name="mu")

    def yhat(self, X: pd.DataFrame | None = None) -> pd.Series:
        """Return the predicted class."""
        if X is None:
            return self.models.model.mu.round(0)
        return self.models.model.predict(X).round(0)

    def yhat_proba(self, X: pd.DataFrame | None = None) -> pd.Series:
        """Return the predicted probability of the positive class."""
        if X is None:
            return self.mu
        return self.models.model.predict(X)

    @property
    def summary(self):
        """Return the summary of the model."""
        return self.models.model.summary()

    @property
    def coefficients(self) -> pd.Series:
        return self.models.model.params

    @property
    def endog(self) -> pd.Series:
        return (
            pd.Series(self.models.model.model.data.endog, name="endog")
            .round(0)
            .astype(int)
        )

    @property
    def exog(self) -> pd.DataFrame:
        return pd.DataFrame(
            self.models.model.model.data.exog, columns=["Intercept"] + self.features
        )

    @property
    def residual_calculator(self) -> BinomialGlmResidualCalculator:
        if self.models.model is None:
            self.fit()

        return BinomialGlmResidualCalculator(
            self.exog, self.endog, self.yhat_proba(), self.coefficients
        )

    @property
    def aic(self) -> float:
        calculator = StatsmodelsGlmAicCalculator(self.models.model)
        return float(calculator.calculate())

    @property
    def deviance(self) -> float:
        calculator = StatsmodelsGlmDevianceCalculator(self.models.model)
        return float(calculator.calculate())

    @property
    def leverage(self) -> pd.Series:
        calculator = BinomialGlmLeverageCalculator(self.exog, self.yhat_proba())
        return calculator.calculate()

    def evaluate_new_feature(
        self, new_feature: str, parallel: bool = True
    ) -> pd.DataFrame:
        evaluator = BinomialGlmAnalysisOfDevianceFeatureEvaluator(
            self, new_feature, parallel
        )
        return evaluator.evaluate_feature()
