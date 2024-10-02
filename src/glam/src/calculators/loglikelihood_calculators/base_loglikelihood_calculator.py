"""Define a protocol for loglikelihood calculators."""

import pandas as pd
from typing import Protocol

__all__ = ["BaseLogLikelihoodCalculator"]


class BaseLogLikelihoodCalculator(Protocol):
    """Protocol for loglikelihood calculators."""

    def calculate(self) -> pd.Series:
        """Calculate the loglikelihood."""
        ...
