"""Concrete implementation of a feature clusterer using hierarchical clustering."""

from __future__ import annotations
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from glam.src.clusterers.feature_clusterers.base_feature_clusterer import BaseFeatureClusterer
from glam.src.data.base_model_data import BaseModelData

class DecisionTreeFeatureClusterer(BaseFeatureClusterer):
    """Concrete implementation of a feature clusterer using hierarchical clustering."""

    def __init__(self, feature: str, data: BaseModelData) -> None:
        """Initialize the clusterer with the number of clusters."""
        super().__init__(feature, data)

    def fit(self, n_clusters: int = 2) -> DecisionTreeFeatureClusterer:
        """Fit the clusterer to the data."""
        # Perform decision tree clustering
        X = self.data.X[self.feature]
        y = self.data.y

        self._clusterer = DecisionTreeClassifier(max_leaf_nodes=n_clusters)
        self._clusterer.fit(X.to_frame(), y)

        self._mapping = {}
        for level in X.unique():
            self._mapping[level] = self._clusterer.predict([[level]])[0]


    def predict(self, X: pd.Series) -> pd.Series:
        """Predict the cluster of each data point."""
        return pd.Series(self._clusterer.predict(X.to_frame()), index=X.index)
