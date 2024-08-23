from typing import Protocol

__all__ = ["BaseBicCalculator"]


class BaseBicCalculator(Protocol):
    """Protocol for BIC calculators."""

    def calculate(self) -> float: ...
