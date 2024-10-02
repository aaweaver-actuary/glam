"""Protocol for likelihood ratio calculators."""

from typing import Protocol


class BaseLikelihoodRatioCalculator(Protocol):
    """Protocol for likelihood ratio calculators."""

    def calculate(self) -> float:
        """Calculate the likelihood ratio."""
        ...
