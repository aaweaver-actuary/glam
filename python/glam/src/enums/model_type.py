"""Module for model task type enum class."""
from enum import StrEnum

__all__ = ["ModelType"]


class ModelType(StrEnum):
    """Enum class for model type."""

    GLM = "glm"
    GEE = "gee"
    GLMM = "glmm"
    GAM = "gam"
    CATBOOST = "catboost"
