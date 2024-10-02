"""Concrete implementation of the BaseDegreesOfFreedomCalculator for GLMs using the statsmodels library."""

from glam.analysis.base_glm_analysis import BaseGlmAnalysis

__all__ = ["StatsmodelsGlmDegreesOfFreedomCalculator"]


class StatsmodelsGlmDegreesOfFreedomCalculator:
    """Concrete implementation of the BaseDegreesOfFreedomCalculator for GLMs using the statsmodels library."""

    def __init__(self, analysis: BaseGlmAnalysis):
        self._analysis = analysis

    @property
    def analysis(self) -> BaseGlmAnalysis:
        """Return the analysis object."""
        return self._analysis

    def calculate(self) -> float:
        """Calculate the degrees of freedom of the GLM."""
        if self.analysis.fitted_model is not None:
            return self.analysis.fitted_model.df_resid
        else:
            raise ValueError("The analysis object does not have a fitted model.")
