"""Concrete implementation of the base feature evaluator using analysis of deviance to evaluate whether a new feature improves a model."""

from __future__ import annotations
from abc import ABC, abstractmethod
from scipy.stats import chi2  # type: ignore
import pandas as pd
from glam.analysis.base_glm_analysis import BaseGlmAnalysis
from glam.src.calculators.deviance_calculators.statsmodels_glm_deviance_calculator import (
    StatsmodelsGlmDevianceCalculator,
)
from glam.src.calculators.aic_calculators.statsmodels_glm_aic_calculator import (
    StatsmodelsGlmAicCalculator,
)
from glam.src.calculators.bic_calculators.statsmodels_glm_bic_calculator import (
    StatsmodelsGlmBicCalculator,
)
from glam.src.calculators.degrees_of_freedom_calculators.degrees_of_freedom_calculator import (
    StatsmodelsGlmDegreesOfFreedomCalculator,
)
import warnings
from statsmodels.tools.sm_exceptions import PerfectSeparationWarning  # type: ignore


class BaseGlmAnalysisOfDevianceFeatureEvaluator(ABC):
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
        parallel: bool = True,
    ):
        self._current_analysis = current_analysis
        self._new_feature = new_feature
        self._deviance_calculator = deviance_calculator
        self._aic_calculator = aic_calculator
        self._bic_calculator = bic_calculator
        self._degrees_of_freedom_calculator = degrees_of_freedom_calculator
        self._parallel = parallel

    @property
    def current_analysis(self) -> BaseGlmAnalysis:
        """Return the current analysis object."""
        return self._current_analysis

    @property
    def new_feature(self) -> str:
        """Return the name of the new feature."""
        return self._new_feature

    @property
    def parallel(self) -> bool:
        """Return whether to use parallel processing."""
        return self._parallel

    @abstractmethod
    def _get_deviance(self, analysis: BaseGlmAnalysis) -> float:
        """Return the deviance of the model."""

    @abstractmethod
    def _get_aic(self, analysis: BaseGlmAnalysis) -> float:
        """Return the AIC of the model."""

    @abstractmethod
    def _get_bic(self, analysis: BaseGlmAnalysis) -> float:
        """Return the BIC of the model."""

    @abstractmethod
    def _get_degrees_of_freedom(self, analysis: BaseGlmAnalysis) -> float:
        """Return the degrees of freedom of the model."""

    def _get_model_with_new_feature(self, analysis: BaseGlmAnalysis) -> BaseGlmAnalysis:
        """Return a new model object with the new feature."""
        warnings.filterwarnings(
            "ignore", category=PerfectSeparationWarning, append=True
        )
        warnings.filterwarnings("ignore", category=RuntimeWarning, append=True)
        cur_features = analysis.features
        cur_data = analysis.data

        if self.new_feature in cur_features:
            return analysis

        new_model = analysis.__class__(cur_data)
        for f in cur_features:
            new_model.add_feature(f)

        new_model.add_feature(self.new_feature)
        new_model.fit(parallel=self.parallel)
        return new_model

    def _get_p_value(
        self, current_model: BaseGlmAnalysis, new_model: BaseGlmAnalysis
    ) -> float:
        """Return the p-value of the new feature."""
        cur_deviance = self._get_deviance(current_model)
        new_deviance = self._get_deviance(new_model)

        cur_dof = self._get_degrees_of_freedom(current_model)
        new_dof = self._get_degrees_of_freedom(new_model)

        deviance_diff = cur_deviance - new_deviance
        dof_diff = cur_dof - new_dof

        return 1 - chi2.cdf(deviance_diff, dof_diff)

    def evaluate_feature(self) -> pd.DataFrame:
        """Evaluate the new feature using analysis of deviance."""
        new_model = self._get_model_with_new_feature(self.current_analysis)

        deviance_current = self._get_deviance(self.current_analysis)
        deviance_new = self._get_deviance(new_model)

        aic_current = self._get_aic(self.current_analysis)
        aic_new = self._get_aic(new_model)

        bic_current = self._get_bic(self.current_analysis)
        bic_new = self._get_bic(new_model)

        dof_current = self._get_degrees_of_freedom(self.current_analysis)
        dof_new = self._get_degrees_of_freedom(new_model)

        p_value = self._get_p_value(self.current_analysis, new_model)

        outdf = pd.DataFrame(
            {
                "Model": [
                    new_model.feature_formula.replace(
                        f"{self.current_analysis.feature_formula}", "[Current Model]"
                    ),
                    self.current_analysis.feature_formula,
                ],
                "Deviance": [deviance_new, deviance_current],
                "AIC": [aic_new, aic_current],
                "BIC": [bic_new, bic_current],
                "DoF": [dof_new, dof_current],
                "pval": [p_value, None],
            }
        ).set_index("Model")

        outdf["Deviance"] = outdf["Deviance"].round(2)
        outdf["AIC"] = outdf["AIC"].round(2)
        outdf["BIC"] = outdf["BIC"].round(2)

        def format_p_value(x: float) -> str:
            if x is None:
                return ""
            elif x < 0.001:
                return f"{x:.1e}"
            else:
                return f"{x:.2f}"

        outdf["p-value"] = outdf["pval"].apply(format_p_value).replace("nan", "")

        return outdf.drop(columns=["pval"])
