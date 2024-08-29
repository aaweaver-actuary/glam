"""Base interface for feature clusterers."""

from __future__ import annotations
from abc import ABC, abstractmethod
import pandas as pd

from glam.src.data.base_model_data import BaseModelData

class BaseFeatureClusterer(ABC):
    """Base interface for feature clusterers."""

    def __init__(self, feature: str, data: BaseModelData) -> None:
        """Initialize the clusterer with the number of clusters."""
        self._feature = feature
        self._data = data
        self._mapping: dict[str, str] | None = None

    def __repr__(self) -> str:
        """Return a string representation of the clusterer."""
        is_fit = self._mapping is not None
        if is_fit:
            return f"{self.__class__.__name__}(feature={self.feature}, mapping={self.mapping})"
        return f"{self.__class__.__name__}(feature={self.feature})"

    def __str__(self) -> str:
        """Return a string representation of the clusterer."""
        return self.__repr__()

    @property
    def feature(self) -> str:
        """Return the feature being clustered."""
        return self._feature

    @property
    def data(self) -> BaseModelData:
        """Return the data being clustered."""
        return self._data

    @property
    def mapping(self) -> dict[str, str] | None:
        """Return the mapping of feature values to cluster labels."""
        if self._mapping is None:
            self.fit()
        return self._mapping

    @abstractmethod
    def fit(self) -> BaseFeatureClusterer:
        """Fit the clusterer to the data."""

    @abstractmethod
    def predict(self, X: pd.Series) -> pd.Series:
        """Predict the cluster of each data point."""
