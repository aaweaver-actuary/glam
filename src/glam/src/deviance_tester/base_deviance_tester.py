import pandas as pd
from typing import Protocol

__all__ = ["BaseDevianceTester"]


class BaseDevianceTester(Protocol):
    def calculate_deviance(self, y: pd.Series, yhat_proba: pd.Series) -> float: ...

    def calculate_aic(self, y: pd.Series, yhat_proba: pd.Series) -> float: ...

    def calculate_deviance_change(
        self, y: pd.Series, yhat_proba: pd.Series, yhat_proba_new: pd.Series
    ) -> float: ...

    def calculate_aic_change(
        self, y: pd.Series, yhat_proba: pd.Series, yhat_proba_new: pd.Series
    ) -> float: ...

    def likelihood_ratio_test(
        self, y: pd.Series, yhat_proba: pd.Series, yhat_proba_new: pd.Series
    ) -> float: ...
