import pandas as pd
from typing import Protocol

class BaseModelFitter(Protocol):
    """Define a protocol interface for fitting a linear model model (GLM, GEE, GLMM, etc.)."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> BaseModelResult: ...
