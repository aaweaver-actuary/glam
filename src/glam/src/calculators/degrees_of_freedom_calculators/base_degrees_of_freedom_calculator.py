"""Define a protocol for degrees of freedom calculators."""

from typing import Protocol

__all__ = ["BaseDegreesOfFreedomCalculator"]


class BaseDegreesOfFreedomCalculator(Protocol):
    """Protocol for deviance calculators."""

    def calculate(self) -> int:
        """Calculate the degrees of freedom of the fitted model."""
        ...
