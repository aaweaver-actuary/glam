"""Module for model task type enum class."""

from enum import StrEnum

__all__ = ["ModelTask"]


class ModelTask(StrEnum):
    """Enum class for model task type."""

    REGRESSION = "regression"
    CLASSIFICATION = "classification"
