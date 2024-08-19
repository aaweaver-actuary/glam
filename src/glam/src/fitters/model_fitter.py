"""This module provides an interface and implementation for fitting a GLM model."""

from typing import Protocol

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from hit_ratio.old_glm.model_result import BaseModelResult




class StatsmodelsGlmFitter:
    """Implement the BaseModelFitter protocol to fit a GLM model using the statsmodels library."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> BaseModelResult:
        model = sm.GLM(y, X, family=sm.families.Binomial())
        result = model.fit()
        return result


class StatsmodelsFormulaGlmFitter:
    """Implement the BaseModelFitter protocol to fit a GLM model using the statsmodels library with a formula."""

    def fit(self, X: pd.DataFrame, y: pd.Series, formula: str) -> BaseModelResult:
        target_name = formula.split("~")[0].strip()
        df = pd.concat([y, X], axis=1)
        df.columns = [target_name] + X.columns.tolist()
        model = smf.glm(formula=formula, data=df, family=sm.families.Binomial())
        result = model.fit()
        return result
