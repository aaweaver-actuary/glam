import pandas as pd
from typing import Protocol

__all__ = ["BaseDevianceCalculator"]


class BaseDevianceCalculator(Protocol):
    """Protocol for deviance calculators."""

    def calculate(self) -> pd.Series: ...
