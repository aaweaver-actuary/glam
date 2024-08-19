"""Define the BaseEvaluator protocol, which is the minimal interface needed to evaluate a model in the GLAM framework."""

from typing import Protocol

__all__ = ["BaseEvaluator"]


class BaseEvaluator(Protocol):
    """Protocol for evaluating models."""

    def evaluate(self) -> None:
        """Evaluate the models."""
        ...
