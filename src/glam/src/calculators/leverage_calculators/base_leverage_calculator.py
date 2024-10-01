import pandas as pd
from typing import Protocol

__all__ = ["BaseLeverageCalculator"]


class BaseLeverageCalculator(Protocol):
    """Protocol for leverage calculators."""

    def calculate(self) -> float | pd.Series: ...
