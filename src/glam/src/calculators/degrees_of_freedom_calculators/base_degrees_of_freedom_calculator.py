from typing import Protocol

__all__ = ["BaseDegreesOfFreedomCalculator"]


class BaseDegreesOfFreedomCalculator(Protocol):
    """Protocol for deviance calculators."""

    def calculate(self) -> int: ...
