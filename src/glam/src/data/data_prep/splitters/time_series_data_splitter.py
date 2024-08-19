import pandas as pd
from glam.src.data.base_model_data import BaseModelData
from typing import Generator

__all__ = ["TimeSeriesDataSplitter"]


class TimeSeriesDataSplitter:
    """Implementation for time-series cross-validation splitting."""

    def __init__(self, data: BaseModelData) -> None:
        self._data = data

    @property
    def data(self) -> BaseModelData:
        """Return the BaseModelData."""
        return self._data

    @property
    def X(self) -> pd.DataFrame:
        """Return the feature DataFrame."""
        return self.data.X

    @property
    def y(self) -> pd.Series:
        """Return the target Series."""
        return self.data.y

    @property
    def cv(self) -> pd.Series:
        """Return the cross-validation labels."""
        return self.data.cv

    @property
    def fold_labels(self) -> list[int]:
        return sorted(self.cv.unique())

    def _validation_train(self, fold: int) -> tuple[pd.DataFrame, pd.Series]:
        train_idx = self.cv < fold
        return self.X[train_idx], self.y[train_idx]

    def _validation_test(self, fold: int) -> tuple[pd.DataFrame, pd.Series]:
        test_idx = self.cv == fold
        return self.X[test_idx], self.y[test_idx]

    def split_data(
        self, fold: int
    ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        train = self._validation_train(fold)
        test = self._validation_test(fold)
        return train[0], train[1], test[0], test[1]

    @property
    def X_y_generator(
        self,
    ) -> Generator[tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series], None, None]:
        for fold in self.fold_labels:
            if fold == 0:
                continue
            yield self.split_data(fold)
