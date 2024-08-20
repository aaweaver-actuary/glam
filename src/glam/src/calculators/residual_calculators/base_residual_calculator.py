import pandas as pd
from typing import Protocol

__all__ = ["BaseResidualCalculator"]


class BaseResidualCalculator(Protocol):
    """Protocol for residual calculators."""

    def deviance_residuals(self) -> pd.Series: ...

    def pearson_residuals(self, std: bool) -> pd.Series: ...

    def anscombe_residuals(self) -> pd.Series: ...

    def partial_residuals(self, feature: str) -> pd.Series: ...
