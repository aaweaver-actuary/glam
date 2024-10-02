"""Define a protocol for leverage calculators."""

from __future__ import annotations
import pandas as pd
from typing import Protocol

__all__ = ["BaseLeverageCalculator"]


class BaseLeverageCalculator(Protocol):
    """Protocol for leverage calculators."""

    def calculate(self) -> float | pd.Series:
        """Calculate the leverage for a single observation or all observations."""
        ...
