import statsmodels.api as sm
import pandas as pd

from glam.src.fitted_model.base_fitted_model import BaseFittedModel


class StatsmodelsGlmFitter:
    """Implement the BaseModelFitter protocol to fit a GLM model using the statsmodels library.

    Attributes
    ----------
    fitted_model : BaseFittedModel | None
        The fitted model.
    """

    def fit(self, X: pd.DataFrame, y: pd.Series) -> BaseFittedModel:
        """Fit a GLM model using the statsmodels library.

        Parameters
        ----------
        X : pd.DataFrame
            The features.
        y : pd.Series
            The target variable.

        Returns
        -------
        BaseFittedModel
            The fitted model.
        """
        model = sm.GLM(y, X, family=sm.families.Binomial())
        return model.fit()
