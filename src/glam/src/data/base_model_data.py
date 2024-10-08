"""Define the BaseData protocol, which is the minimal interface needed to load and prepare data in the GLAM framework."""

from __future__ import annotations
from typing import Protocol
import pandas as pd
import polars as pl

__all__ = ["BaseModelData"]


class BaseModelData(Protocol):
    """Class providing an interface for the GLM data functionality we need, no matter the underlying library."""

    @property
    def lf(self) -> pl.LazyFrame:
        """Return the full data frame as a polars LazyFrame."""
        ...

    @property
    def df(self) -> pd.DataFrame:
        """Return the full data frame."""
        ...

    @property
    def X(self) -> pd.DataFrame:
        """Return the full feature (X) matrix."""
        ...

    @property
    def y(self) -> pd.Series:
        """Return the response (y) vector."""
        ...

    @property
    def feature_names(self) -> list[str]:
        """Return the names of all possible features in the data."""
        ...

    @property
    def cv(self) -> pd.Series:
        """Return the cross-validation fold assignments.

        Training folds are typically labeled with non-negative integers, while validation and testing folds are labeled with 'val' and 'test', respectively.
        """
        ...

    @property
    def unanalyzed(self) -> list[str]:
        """Return the names of the unanalyzed features."""
        ...

    def add_feature(self, name: str, values: pd.Series) -> None:
        """Add a new feature to the data.

        Used (for example) when creating dummy variables, interaction terms, or polynomial features.
        """
        ...
