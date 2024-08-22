"""Concrete implementation of the BaseDegreesOfFreedomCalculator for GLMs using the statsmodels library."""

from glam.analysis.base_analysis import BaseAnalysis

__all__ = ["DegreesOfFreedomCalculator"]


class DegreesOfFreedomCalculator:
    """Concrete implementation of the BaseDegreesOfFreedomCalculator for GLMs using the statsmodels library."""

    def __init__(self, analysis: BaseAnalysis):
        self._analysis = analysis

    @property
    def analysis(self) -> BaseAnalysis:
        return self._analysis

    def calculate(self) -> float:
        """Calculate the degrees of freedom of the GLM."""
        return self.analysis.models.model.df_resid
