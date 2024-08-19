from typing import Protocol, List, Tuple
import pandas as pd


class BaseData(Protocol):
    """Protocol for data functionality."""

    @property
    def X(self) -> pd.DataFrame:
        """Return the feature matrix."""
        ...

    @property
    def y(self) -> pd.Series:
        """Return the response variable."""
        ...

    @property
    def feature_names(self) -> List[str]:
        """Return the names of the features."""
        ...

    @property
    def cv(self) -> pd.Series:
        """Return the cross-validation fold."""
        ...

    @property
    def unanalyzed(self) -> List[str]:
        """Return the names of the unanalyzed features."""
        ...


class GlmData:
    """Concrete implementation of the data functionality using pandas."""

    def __init__(
        self,
        df: pd.DataFrame,
        y: str,
        cv: str = "fold",
        unanalyzed: List[str] = None,
    ):
        self._df = df
        self._y_col = y
        self._cv_col = cv
        self._unanalyzed = unanalyzed if unanalyzed else []

        self._X = self._df.drop(columns=[self._y_col, self._cv_col] + self._unanalyzed)
        self._y = self._df[self._y_col]
        self._cv = self._df[self._cv_col]

        if splitter:
            self._splitter = splitter
        else:
            self._splitter = TimeSeriesDataSplitter(
                self._X, self._y, self._cv
            )  # Default splitter

    @property
    def X(self) -> pd.DataFrame:
        return self._X

    @property
    def y(self) -> pd.Series:
        return self._y

    @property
    def feature_names(self) -> List[str]:
        return self._X.columns.to_list()

    @property
    def cv(self) -> pd.Series:
        return self._cv

    @property
    def unanalyzed(self) -> List[str]:
        return self._unanalyzed

    def X_y_generator(self) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        for fold in self._splitter.fold_labels:
            yield self._splitter.split_data(fold)
