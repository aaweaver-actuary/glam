"""Concrete implementation of the BaseGlmAnalysis class for binomial GLM analysis."""

from __future__ import annotations
import pandas as pd
import logging

import statsmodels.api as sm  # type: ignore
from glam.analysis.base_glm_analysis import BaseGlmAnalysis
from glam.src.calculators.deviance_calculators.statsmodels_glm_deviance_calculator import (
    StatsmodelsGlmDevianceCalculator,
)
from glam.src.data.base_model_data import BaseModelData
from glam.src.enums.model_task import ModelTask
from glam.src.fitters.statsmodels_formula_glm_fitter import StatsmodelsFormulaGlmFitter
from glam.src.model_list.base_model_list import BaseModelList
from glam.src.fitted_model.statsmodels_fitted_glm import StatsmodelsFittedGlm
from glam.src.data.data_prep import BaseDataSplitter, BasePreprocessor
from glam.src.model_list.default_model_list import DefaultModelList
from glam.src.data.data_prep import TimeSeriesDataSplitter, DefaultPreprocessor
from glam.src.calculators.residual_calculators.binomial_glm_residual_calculator import (
    BinomialGlmResidualCalculator,
)
from glam.src.calculators.aic_calculators.statsmodels_glm_aic_calculator import (
    StatsmodelsGlmAicCalculator,
)
from glam.src.calculators.bic_calculators.statsmodels_glm_bic_calculator import (
    StatsmodelsGlmBicCalculator,
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


class BinomialGlmAnalysis(BaseGlmAnalysis):
    """Class for binomial GLM analysis."""

    def __init__(
        self,
        data: BaseModelData,
        fitter: StatsmodelsFormulaGlmFitter | None = None,
        models: BaseModelList | None = None,
        features: list[str] | None = None,
        interactions: list[str] | None = None,
        fitted_model: StatsmodelsFittedGlm | None = None,
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
        self._fitter = fitter if fitter is not None else StatsmodelsFormulaGlmFitter()  # type: ignore
        self._models = models if models is not None else DefaultModelList()
        self._fitted_model = (
            fitted_model if fitted_model is not None else StatsmodelsFittedGlm()  # type: ignore
        )
        self._features = features if features is not None else []
        self._interactions = interactions if interactions is not None else []

        self._splitter = (
            splitter if splitter is not None else TimeSeriesDataSplitter(data)
        )
        self._preprocessor = (
            preprocessor if preprocessor is not None else DefaultPreprocessor(data)  # type: ignore
        )

        self._task = task

    def __repr__(self):
        """Return the string representation of the class."""
        if len(self.features) > 0:
            return f"BinaryGlmAnalysis({self.linear_formula})"

        return f"BinaryGlmAnalysis({self.data.y.name} ~ 1)"

    @property
    def model(self) -> sm.genmod.generalized_linear_model.GLMResults:
        """Return the fitted model."""
        if self._fitted_model is None:
            self.fit()

        return self._fitted_model

    @property
    def mu(self) -> pd.Series:
        """Return the expected value of the response variable."""
        return pd.Series(self.model.mu, name="mu")

    def yhat(self, X: pd.DataFrame | None = None) -> pd.Series:
        """Return the predicted class."""
        if X is None:
            return self.model.mu.round(0)
        return self.model.predict(X).round(0)

    def yhat_proba(self, X: pd.DataFrame | None = None) -> pd.Series:
        """Return the predicted probability of the positive class."""
        if X is None:
            return self.mu
        return self.model.predict(X)

    @property
    def summary(self) -> pd.DataFrame:
        """Return the summary of the model."""
        return self.model.summary()

    @property
    def coefficients(self) -> pd.Series:
        """Return the coefficients of the model."""
        return self.model.params

    @property
    def endog(self) -> pd.Series:
        """Return the endogenous variable."""
        return pd.Series(self.model.model.data.endog, name="endog").round(0).astype(int)

    @property
    def exog(self) -> pd.DataFrame:
        """Return the exogenous variables."""
        return pd.DataFrame(
            self.model.model.data.exog, columns=["Intercept", *self.features]
        )

    @property
    def residual_calculator(self) -> BinomialGlmResidualCalculator:  # type: ignore
        """Return the residual calculator."""
        _ = self.model
        return BinomialGlmResidualCalculator(
            self.exog, self.endog, self.yhat_proba(), self.coefficients
        )

    @property
    def aic(self) -> float:
        """Return the AIC of the model."""
        calculator = StatsmodelsGlmAicCalculator(self.model)
        return float(calculator.calculate())

    @property
    def bic(self) -> float:
        """Return the BIC of the model."""
        calculator = StatsmodelsGlmBicCalculator(self.model)
        return float(calculator.calculate())

    @property
    def deviance(self) -> float:
        """Calculate and return the deviance of the model."""
        calculator = StatsmodelsGlmDevianceCalculator(self.model)
        return float(calculator.calculate())

    @property
    def leverage(self) -> pd.Series:
        """Calculate and return the leverage of the model."""
        calculator = BinomialGlmLeverageCalculator(self.exog, self.yhat_proba())
        return calculator.calculate_all()

    def evaluate_new_feature(
        self, new_feature: str, parallel: bool = False
    ) -> pd.DataFrame:
        """Evaluate a new feature."""
        evaluator = BinomialGlmAnalysisOfDevianceFeatureEvaluator(
            self, new_feature, parallel=parallel
        )
        return evaluator.evaluate_feature()
