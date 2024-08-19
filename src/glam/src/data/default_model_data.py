import pandas as pd

__all__ = ["DefaultModelData"]


class DefaultModelData:
    """This class provides a concrete implementation of the model data functionality using pandas."""

    def __init__(
        self,
        df: pd.DataFrame,
        y: str | None = None,
        cv: str | None = None,
        unanalyzed: str | list[str] | None = None,
        is_time_series_cv: bool = False,
    ):
        self._df = df
        self._y = y if y is not None else df.columns[-1]
        self._cv = cv if cv is not None else "fold"
        self._unanalyzed = unanalyzed if unanalyzed is not None else []

    @property
    def df(self) -> pd.DataFrame:
        """Return the data frame."""
        return self._df

    @df.setter
    def df(self, df: pd.DataFrame) -> None:
        """Set the data frame."""
        self._df = df

    @property
    def X(self) -> pd.DataFrame:
        """Return the feature matrix."""
        return self._df.drop(columns=[self._y, self._cv] + self._unanalyzed)

    @property
    def y(self) -> pd.Series:
        """Return the response variable."""
        return self.df[self._y]

    @property
    def feature_names(self) -> list[str]:
        """Return the names of the features."""
        return self.X.columns.tolist()

    @property
    def cv(self) -> pd.Series:
        """Return the cross-validation fold."""
        return self._df[self._cv]

    @property
    def unanalyzed(self) -> list[str]:
        """Return the names of the unanalyzed features."""
        return self._unanalyzed

    @unanalyzed.setter
    def unanalyzed(self, unanalyzed: list[str]) -> None:
        """Set the names of the unanalyzed features."""
        self._unanalyzed = unanalyzed
