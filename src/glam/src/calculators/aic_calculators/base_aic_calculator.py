"""Define a protocol for AIC calculators."""

from typing import Protocol

__all__ = ["BaseAicCalculator"]


class BaseAicCalculator(Protocol):
    """Protocol for AIC calculators."""

    def calculate(self) -> float:
        """Calculate the AIC of the fitted model."""
        ...
