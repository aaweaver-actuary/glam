"""Concrete implementation of a feature clusterer using scikit-learn's implementation of KNN."""

from __future__ import annotations
import pandas as pd
from sklearn.cluster import KMeans

from glam.src.clusterers.feature_clusterers.base_feature_clusterer import BaseFeatureClusterer
from glam.src.data.base_model_data import BaseModelData

class SklearnKnnFeatureClusterer(BaseFeatureClusterer):
    """Concrete implementation of a feature clusterer using scikit-learn's implementation of KNN."""

    def __init__(self, n_clusters: int, feature: str, data: BaseModelData) -> None:
        """Initialize the clusterer with the number of clusters."""
        super().__init__(n_clusters, feature, data)
        self._clusterer: KMeans | None = None

    def fit(self) -> SklearnKnnFeatureClusterer:
        """Fit the clusterer to the data."""
        self._clusterer = KMeans(n_clusters=self.n_clusters)
        self._clusterer.fit(self.data.df[[self.feature]])
        self._mapping = {
            value: f"Cluster {label}"
            for value, label in zip(
                self.data.df[self.feature].unique(),
                self._clusterer.predict(self.data.df[[self.feature]].drop_duplicates()),
            )
        }
        return self

    def predict(self, X: pd.Series) -> pd.Series:
        """Predict the cluster of each data point."""
        return pd.Series(self._clusterer.predict(X.to_frame()), index=X.index)
