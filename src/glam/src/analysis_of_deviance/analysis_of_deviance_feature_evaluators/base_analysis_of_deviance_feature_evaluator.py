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
from glam.src.calculators.aic_calculators.base_aic_calculator import BaseAicCalculator
from glam.src.calculators.bic_calculators.base_bic_calculator import BaseBicCalculator
import warnings
from statsmodels.tools.sm_exceptions import PerfectSeparationWarning


class BaseAnalysisOfDevianceFeatureEvaluator(ABC):
    """Concrete implementation of the base feature evaluator using analysis of deviance to evaluate whether a new feature improves a model."""

    def __init__(
        self,
        current_analysis: BaseAnalysis,
        new_feature: str,
        deviance_calculator: BaseDevianceCalculator,
        aic_calculator: BaseAicCalculator,
        bic_calculator: BaseBicCalculator,
        degrees_of_freedom_calculator: BaseDegreesOfFreedomCalculator,
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
    def current_analysis(self):
        return self._current_analysis

    @property
    def new_feature(self):
        return self._new_feature

    @property
    def parallel(self):
        return self._parallel

    @abstractmethod
    def _get_deviance(self, analysis: BaseAnalysis) -> float:
        pass

    @abstractmethod
    def _get_aic(self, analysis: BaseAnalysis) -> float:
        pass

    @abstractmethod
    def _get_bic(self, analysis: BaseAnalysis) -> float:
        pass

    @abstractmethod
    def _get_degrees_of_freedom(self, analysis: BaseAnalysis) -> float:
        pass

    def _get_model_with_new_feature(self, analysis: BaseAnalysis) -> BaseAnalysis:
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

        def format_p_value(x):
            if x is None:
                return ""
            elif x < 0.001:
                return f"{x:.1e}"
            else:
                return f"{x:.2f}"

        outdf["p-value"] = outdf["pval"].apply(format_p_value).replace("nan", "")

        return outdf.drop(columns=["pval"])
