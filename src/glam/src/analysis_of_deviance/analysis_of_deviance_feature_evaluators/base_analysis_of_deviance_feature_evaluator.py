"""Concrete implementation of the base feature evaluator using analysis of deviance to evaluate whether a new feature improves a model."""

from abc import ABC, abstractmethod
from scipy.stats import chi2
import pandas as pd
from glam.analysis.base_analysis import BaseAnalysis
from glam.src.calculators.deviance_calculators.base_deviance_calculator import (
    BaseDevianceCalculator,
)
from glam.src.calculators.degrees_of_freedom_calculators.base_degrees_of_freedom_calculator import (
    BaseDegreesOfFreedomCalculator,
)


class BaseAnalysisOfDevianceFeatureEvaluator(ABC):
    """Concrete implementation of the base feature evaluator using analysis of deviance to evaluate whether a new feature improves a model."""

    def __init__(
        self,
        current_analysis: BaseAnalysis,
        new_feature: str,
        deviance_calculator: BaseDevianceCalculator,
        degrees_of_freedom_calculator: BaseDegreesOfFreedomCalculator,
    ):
        self._current_analysis = current_analysis
        self._new_feature = new_feature
        self._deviance_calculator = deviance_calculator
        self._degrees_of_freedom_calculator = degrees_of_freedom_calculator

    @property
    def current_analysis(self):
        return self._current_analysis

    @property
    def new_feature(self):
        return self._new_feature

    @abstractmethod
    def _get_deviance(self, analysis: BaseAnalysis) -> float:
        pass

    @abstractmethod
    def _get_degrees_of_freedom(self, analysis: BaseAnalysis) -> float:
        pass

    def _get_model_with_new_feature(self, analysis: BaseAnalysis) -> BaseAnalysis:
        new_model = analysis.copy()
        new_model.add_feature(self.new_feature)
        new_model.fit()
        return new_model

    def _get_p_value(
        self, current_model: BaseAnalysis, new_model: BaseAnalysis
    ) -> float:
        cur_deviance = self._get_deviance(current_model)
        new_deviance = self._get_deviance(new_model)

        cur_dof = self._get_degrees_of_freedom(current_model)
        new_dof = self._get_degrees_of_freedom(new_model)

        deviance_diff = cur_deviance - new_deviance
        dof_diff = cur_dof - new_dof

        return 1 - chi2.cdf(deviance_diff, dof_diff)

    def evaluate_feature(self) -> pd.DataFrame:
        """Evaluate the new feature using analysis of deviance."""
        # New model object with the new feature
        new_model = self._get_model_with_new_feature(self.current_analysis)

        deviance_current = self._get_deviance(self.current_analysis)
        deviance_new = self._get_deviance(new_model)

        dof_current = self._get_degrees_of_freedom(self.current_analysis)
        dof_new = self._get_degrees_of_freedom(new_model)

        p_value = self._get_p_value(self.current_analysis, new_model)

        outdf = pd.DataFrame(
            {
                "Model": [
                    new_model.feature_formula,
                    self.current_analysis.feature_formula,
                ],
                "Deviance": [deviance_new, deviance_current],
                "DoF": [dof_new, dof_current],
                "pval": [p_value, None],
            }
        ).set_index("Model")

        outdf["Deviance"] = outdf["Deviance"].round(2)

        def format_p_value(x):
            if x is None:
                return ""
            elif x < 0.001:
                return f"{x:.1e}"
            else:
                return f"{x:.2f}"

        outdf["p-value"] = outdf["pval"].apply(format_p_value).replace("nan", "")

        return outdf.drop(columns=["pval"])
