from typing import Protocol


class OneWay(Protocol):
    """Protocol for one-way ANOVA."""

    @property
    def model(self) -> str:
        """Return the model formula."""
        ...

    def f_test(self) -> tuple[float, float, float, float]:
        """Perform an F-test."""
        ...
