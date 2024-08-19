from typing import Protocol
import pandas as pd
from hit_ratio.old_glm.model_data import BaseModelData


class BaseDataSplitter(Protocol):
    """Protocol for splitting data."""

    def split_data(
        self, fold: int
    ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """Split the data into training and testing sets for the given fold."""
        ...

    @property
    def X_y_generator(self) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """Generate train-test splits for cross-validation."""
        ...

    @property
    def fold_labels(self) -> list[int]:
        """Return the list of unique fold labels."""
        ...


class TimeSeriesDataSplitter:
    """Implementation for time-series cross-validation splitting."""

    def fold_labels(self, data: BaseModelData) -> list[int]:
        return sorted(data.cv.unique())

    def _validation_train(
        self, fold: int, data: BaseModelData
    ) -> tuple[pd.DataFrame, pd.Series]:
        train_idx = data.cv < fold
        return data.X[train_idx], data.y[train_idx]

    def _validation_test(
        self, fold: int, data: BaseModelData
    ) -> tuple[pd.DataFrame, pd.Series]:
        test_idx = data.cv == fold
        return data.X[test_idx], data.y[test_idx]

    def split_data(
        self, fold: int, data: BaseModelData
    ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        train = self._validation_train(fold, data)
        test = self._validation_test(fold, data)
        return train[0], train[1], test[0], test[1]

    def X_y_generator(
        self, data: BaseModelData
    ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        for fold in self.fold_labels(data):
            if fold == 0:
                continue
            yield self.split_data(fold, data)


class DefaultDataSplitter:
    """Implementation for normal cross-validation splitting."""

    def fold_labels(self, data: BaseModelData) -> list[int]:
        return sorted(data.cv.unique())

    def _validation_train(
        self, fold: int, data: BaseModelData
    ) -> tuple[pd.DataFrame, pd.Series]:
        train_idx = data.cv != fold
        return data.X[train_idx], data.y[train_idx]

    def _validation_test(
        self, fold: int, data: BaseModelData
    ) -> tuple[pd.DataFrame, pd.Series]:
        test_idx = data.cv == fold
        return data.X[test_idx], data.y[test_idx]

    def split_data(
        self, fold: int, data: BaseModelData
    ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        train = self._validation_train(fold, data)
        test = self._validation_test(fold, data)
        return train[0], train[1], test[0], test[1]

    def X_y_generator(
        self, data: BaseModelData
    ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        for fold in self.fold_labels:
            yield self.split_data(fold, data)
