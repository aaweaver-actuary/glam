"""Module providing the DefaultModelData class for handling model data using pandas."""

from __future__ import annotations
import pandas as pd

__all__ = ["DefaultModelData"]


class DefaultModelData:
    """Class providing a concrete implementation of the model data functionality using pandas."""

    def __init__(
        self,
        df: pd.DataFrame,
        y: str | None = None,
        cv: str | None = None,
        unanalyzed: str | list[str] | None = None,
        is_time_series_cv: bool = True,
    ):
        self._df = df
        self._y = y if y is not None else df.columns.tolist()[-1]
        self._cv = cv if cv is not None else "fold"
        self._unanalyzed = unanalyzed if unanalyzed is not None else []
        self._is_time_series_cv = is_time_series_cv

    def __repr__(self) -> str:
        """Return a string representation of the DefaultModelData object.

        Returns
        -------
        str
            String representation of the DefaultModelData object.
        """
        return f"DefaultModelData(y='{self._y}', cv='{self._cv}', df.shape={self._df.shape})"

    def __str__(self) -> str:
        """Return a string representation of the object."""
        return self.__repr__()

    @property
    def df(self) -> pd.DataFrame:
        """Return the data frame."""
        return self._df

    @df.setter
    def df(self, df: pd.DataFrame) -> None:
        """Set the data frame.

        Parameters
        ----------
        df : pd.DataFrame
            The new data frame.
        """
        self._df = df

    @property
    def X(self) -> pd.DataFrame:
        """Return the feature matrix."""
        return self._df.drop(columns=[self._y, self._cv, *self._unanalyzed])

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
        """Set the names of the unanalyzed features.

        Parameters
        ----------
        unanalyzed : list[str]
            The names of the unanalyzed features.
        """
        self._unanalyzed = unanalyzed

    @property
    def is_time_series_cv(self) -> bool:
        """Return whether the cross-validation is time series."""
        return self._is_time_series_cv

    def add_feature(self, name: str, values: pd.Series) -> None:
        """Add a new feature to the DataFrame.

        Parameters
        ----------
        name : str
            The name of the new feature.
        values : pd.Series
            The values of the new feature.

        Returns
        -------
        None
        """
        self.df = pd.concat([self._df, pd.Series(values, name=name)], axis=1)
