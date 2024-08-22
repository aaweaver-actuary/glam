from glam.analysis.base_analysis import BaseAnalysis
from glam.src.calculators.deviance_calculators.statsmodels_glm_deviance_calculator import (
    StatsmodelsGlmDevianceCalculator,
)
from glam.src.analysis_of_deviance.analysis_of_deviance_feature_evaluators.base_analysis_of_deviance_feature_evaluator import (
    BaseAnalysisOfDevianceFeatureEvaluator,
)
from glam.src.calculators.degrees_of_freedom_calculators.degrees_of_freedom_calculator import (
    DegreesOfFreedomCalculator,
)


class BinomialGlmAnalysisOfDevianceFeatureEvaluator(
    BaseAnalysisOfDevianceFeatureEvaluator
):
    """Concrete implementation of the base feature evaluator using analysis of deviance to evaluate whether a new feature improves a model."""

    def __init__(
        self,
        current_analysis: BaseAnalysis,
        new_feature: str,
        deviance_calculator: StatsmodelsGlmDevianceCalculator | None = None,
        degrees_of_freedom_calculator: DegreesOfFreedomCalculator | None = None,
    ):
        super().__init__(
            current_analysis,
            new_feature,
            deviance_calculator,
            degrees_of_freedom_calculator,
        )

    @property
    def deviance_calculator(self):
        return (
            self._deviance_calculator
            if hasattr(self, "_deviance_calculator")
            else StatsmodelsGlmDevianceCalculator
        )

    @property
    def degrees_of_freedom_calculator(self):
        return (
            self._degrees_of_freedom_calculator
            if hasattr(self, "_degrees_of_freedom_calculator")
            else DegreesOfFreedomCalculator
        )

    def _get_deviance(self, analysis: BaseAnalysis) -> float:
        calculator = (
            self.deviance_calculator(analysis)
            if self.deviance_calculator is not None
            else StatsmodelsGlmDevianceCalculator(analysis)
        )
        return calculator.calculate()

    def _get_degrees_of_freedom(self, analysis: BaseAnalysis) -> float:
        calculator = (
            self._degrees_of_freedom_calculator(analysis)
            if self._degrees_of_freedom_calculator is not None
            else DegreesOfFreedomCalculator(analysis)
        )
        return calculator.calculate()
