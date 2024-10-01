"""Defines the protocol for a feature evaluator."""

from typing import Protocol
import pandas as pd

__all__ = ["BaseFeatureEvaluator"]


class BaseFeatureEvaluator(Protocol):
    """Protocol for feature evaluators.

    The protocol requires a single method, evaluate_feature, which takes a new feature as
    input and returns a DataFrame with the results of evaluating the feature.
    """

    def evaluate_feature(self, new_feature: str) -> pd.DataFrame: ...
