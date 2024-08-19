from enum import StrEnum

__all__ = ["ModelTask"]


class ModelTask(StrEnum):
    REGRESSION = "regression"
    CLASSIFICATION = "classification"
