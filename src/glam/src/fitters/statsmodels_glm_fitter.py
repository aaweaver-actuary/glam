import statsmodels.api as sm
import pandas as pd

from glam.src.fitted_model.base_fitted_model import BaseFittedModel

class StatsmodelsGlmFitter:
    """Implement the BaseModelFitter protocol to fit a GLM model using the statsmodels library."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> BaseFittedModel:
        model = sm.GLM(y, X, family=sm.families.Binomial())
        return model.fit()
        