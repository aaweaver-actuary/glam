"""Define a protocol for residual calculators."""

import pandas as pd
from typing import Protocol

__all__ = ["BaseResidualCalculator"]


class BaseResidualCalculator(Protocol):
    """Protocol for residual calculators."""

    def deviance_residuals(self) -> pd.Series:
        """Return the deviance residuals."""
        ...

    def pearson_residuals(self, std: bool) -> pd.Series:
        """Return the Pearson residuals."""
        ...

    def anscombe_residuals(self) -> pd.Series:
        """Return the Anscombe residuals."""
        ...

    def partial_residuals(self, feature: str) -> pd.Series:
        """Return the partial residuals for a given feature."""
        ...
