import numpy as np
import pandas as pd
from glam.src.resid.base_deviance_resid import BaseDevianceResid

__all__ = ["BinomialDevianceResid"]


class BinomialDevianceResid(BaseDevianceResid):
    """Calculate deviance residuals for a fitted binomial GLM."""

    def log_likelihood(self, y: pd.Series, yhat_proba: pd.Series) -> pd.Series:
        """For a binomial GLM, the log-likelihood is the binomial log-likelihood:

        log-likelihood = y * log(yhat_proba) + (1 - y) * log(1 - yhat_proba)

        where y is the response variable and yhat_proba is the predicted probability.
        """
        return (y * yhat_proba.apply(np.log)) + (
            (1 - y) * (1 - yhat_proba).apply(np.log)
        )
