"""Define the BinomialGlmAnalysisOfDevianceFeatureEvaluator class."""

from __future__ import annotations
from glam.analysis.base_glm_analysis import BaseGlmAnalysis
from glam.src.calculators.deviance_calculators.statsmodels_glm_deviance_calculator import (
    StatsmodelsGlmDevianceCalculator,
)
from glam.src.analysis_of_deviance.analysis_of_deviance_feature_evaluators.base_analysis_of_deviance_feature_evaluator import (
    BaseGlmAnalysisOfDevianceFeatureEvaluator,
)
from glam.src.calculators.degrees_of_freedom_calculators.degrees_of_freedom_calculator import (
    StatsmodelsGlmDegreesOfFreedomCalculator,
)
from glam.src.calculators.aic_calculators.statsmodels_glm_aic_calculator import (
    StatsmodelsGlmAicCalculator,
)
from glam.src.calculators.bic_calculators.statsmodels_glm_bic_calculator import (
    StatsmodelsGlmBicCalculator,
)


class BinomialGlmAnalysisOfDevianceFeatureEvaluator(
    BaseGlmAnalysisOfDevianceFeatureEvaluator
):
    """Concrete implementation of the base feature evaluator using analysis of deviance to evaluate whether a new feature improves a model."""

    def __init__(
        self,
        current_analysis: BaseGlmAnalysis,
        new_feature: str,
        deviance_calculator: StatsmodelsGlmDevianceCalculator | None = None,
        aic_calculator: StatsmodelsGlmAicCalculator | None = None,
        bic_calculator: StatsmodelsGlmBicCalculator | None = None,
        degrees_of_freedom_calculator: StatsmodelsGlmDegreesOfFreedomCalculator
        | None = None,
        parallel: bool = False,
    ):
        super().__init__(
            current_analysis,
            new_feature,
            deviance_calculator,
            aic_calculator,
            bic_calculator,
            degrees_of_freedom_calculator,
            parallel,
        )

    @property
    def deviance_calculator(self) -> StatsmodelsGlmDevianceCalculator:
        """Return the deviance calculator."""
        if self._deviance_calculator is None:
            self._deviance_calculator = StatsmodelsGlmDevianceCalculator(
                self.current_analysis.fitted_model
            )
        return self._deviance_calculator

    @property
    def aic_calculator(self) -> StatsmodelsGlmAicCalculator:
        """Return the AIC calculator."""
        if self._aic_calculator is None:
            self._aic_calculator = StatsmodelsGlmAicCalculator(self.current_analysis)
        return self._aic_calculator

    @property
    def bic_calculator(self) -> StatsmodelsGlmBicCalculator:
        """Return the BIC calculator object."""
        if self._bic_calculator is None:
            self._bic_calculator = StatsmodelsGlmBicCalculator(self.current_analysis)
        return self._bic_calculator

    @property
    def degrees_of_freedom_calculator(self) -> StatsmodelsGlmDegreesOfFreedomCalculator:
        """Return the degrees of freedom calculator object."""
        if self._degrees_of_freedom_calculator is None:
            self._degrees_of_freedom_calculator = (
                StatsmodelsGlmDegreesOfFreedomCalculator(self.current_analysis)
            )
        return self._degrees_of_freedom_calculator

    def _get_deviance(self, analysis: BaseGlmAnalysis) -> float:
        """Return the deviance of the model."""
        calculator = StatsmodelsGlmDevianceCalculator(analysis)
        return calculator.calculate()

    def _get_aic(self, analysis: BaseGlmAnalysis) -> float:
        """Return the AIC of the model."""
        calculator = StatsmodelsGlmAicCalculator(analysis)
        return calculator.calculate()

    def _get_bic(self, analysis: BaseGlmAnalysis) -> float:
        """Return the BIC of the model."""
        calculator = StatsmodelsGlmBicCalculator(analysis)
        return calculator.calculate()

    def _get_degrees_of_freedom(self, analysis: BaseGlmAnalysis) -> float:
        """Return the degrees of freedom of the model."""
        calculator = StatsmodelsGlmDegreesOfFreedomCalculator(analysis)
        return calculator.calculate()
