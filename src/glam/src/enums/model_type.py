from enum import StrEnum

__all__ = ["ModelType"]


class ModelType(StrEnum):
    GLM = "glm"
    GEE = "gee"
    GLMM = "glmm"
    GAM = "gam"
    CATBOOST = "catboost"
