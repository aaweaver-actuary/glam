"""Module providing the DefaultModelData class for handling model data using pandas."""

from __future__ import annotations
import polars as pl
import pandas as pd
import numpy as np

__all__ = ["DefaultModelData"]


class DefaultModelData:
    """Class providing a concrete implementation of the model data functionality using pandas."""

    def __init__(
        self,
        df: pd.DataFrame | pl.DataFrame | pl.LazyFrame,
        y: str | None = None,
        cv: str | None = None,
        unanalyzed: str | list[str] | None = None,
        is_time_series_cv: bool = True,
    ):
        # Convert the DataFrame to a polars LazyFrame
        if isinstance(df, pd.DataFrame):
            lf = pl.from_pandas(df).lazy()
        elif isinstance(df, pl.DataFrame):
            lf = df.lazy()
        elif isinstance(df, pl.LazyFrame):
            lf = df
        else:
            raise ValueError(
                "DataFrame must be a pandas DataFrame or a polars DataFrame/LazyFrame."
            )
        # Raise an error if the DataFrame is empty
        if lf.collect().shape[0] == 0:
            raise ValueError("DataFrame cannot be empty.")

        # Data are stored as polars LazyFrames and only materialized when needed
        self._df = lf

        if (y not in lf.collect_schema().names()) and (
            "target" not in lf.collect_schema().names()
        ):
            raise KeyError(f"Response variable '{y}' not found in the DataFrame.")
        self._y = y if y is not None else lf.collect_schema().names()[-1]

        if (cv not in lf.collect_schema().names()) and (
            "fold" not in lf.collect_schema().names()
        ):
            raise KeyError(
                f"Cross-validation fold column '{cv}' not found in the DataFrame."
            )

        self._cv = cv if cv is not None else "fold"
        self._unanalyzed = unanalyzed if unanalyzed is not None else []
        self._is_time_series_cv = is_time_series_cv
        self._shape = lf.collect().shape

    def __repr__(self) -> str:
        """Return a string representation of the DefaultModelData object.

        Returns
        -------
        str
            String representation of the DefaultModelData object.
        """
        return f"ModelData(y='{self._y}', cv='{self._cv}', shape={self.shape})"

    def __str__(self) -> str:
        """Return a string representation of the object."""
        return self.__repr__()

    @property
    def shape(self) -> tuple[int, int]:
        """Return the shape of the data frame."""
        return self._shape

    @property
    def lf(self) -> pl.LazyFrame:
        """Return the polars LazyFrame."""
        return self._df

    @lf.setter
    def lf(self, lf: pl.LazyFrame) -> None:
        """Set the polars LazyFrame.

        Parameters
        ----------
        lf : pl.LazyFrame
            The new polars LazyFrame.
        """
        self._df = lf

    @property
    def df(self) -> pd.DataFrame:
        """Return the data frame."""
        return self._df.collect().to_pandas()

    @df.setter
    def df(self, df: pd.DataFrame) -> None:
        """Set the data frame.

        Parameters
        ----------
        df : pd.DataFrame
            The new data frame.
        """
        self._df = pl.from_pandas(df).lazy()

    @property
    def df_cols(self) -> list[str]:
        """Return the columns of the data frame."""
        return self._df.collect_schema().names()

    @property
    def X(self) -> pd.DataFrame:
        """Return the feature matrix."""
        return (
            self._df.drop([self._y, self._cv, *self._unanalyzed]).collect().to_pandas()
        )

    @property
    def y(self) -> pd.Series:
        """Return the response variable."""
        if self._y in self._unanalyzed:
            raise ValueError(
                f"Response variable '{self._y}' is in the unanalyzed features."
            )

        return self._df.select([self._y]).collect().to_pandas()[self._y]

    @property
    def feature_names(self) -> list[str]:
        """Return the names of the features."""
        return self.X.columns.tolist()

    @property
    def cv(self) -> pd.Series:
        """Return the cross-validation fold."""
        return self._df.select([self._cv]).collect().to_pandas()[self._cv]

    @property
    def unanalyzed(self) -> list[str]:
        """Return the names of the unanalyzed features."""
        return (
            self._unanalyzed
            if isinstance(self._unanalyzed, list)
            else [self._unanalyzed]
        )

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

    def add_feature(
        self, name: str, values: pl.Series | pd.Series | list | np.ndarray
    ) -> None:
        """Add a new feature to the DataFrame.

        Parameters
        ----------
        name : str
            The name of the new feature.
        values : pl.Series | pd.Series | list | np.ndarray
            The values of the new feature.

        Returns
        -------
        None
        """
        if name in self.df_cols:
            raise ValueError(f"Feature '{name}' already exists in the DataFrame")

        if len(values) != self.shape[0]:
            raise ValueError(
                f"Length of new feature '{name}' ({len(values)}) does not match the number of rows in the DataFrame ({self.shape[0]})"
            )

        if isinstance(values, (pd.Series, list, np.ndarray)):
            self.lf = self._df.with_columns(pl.Series(values))
        elif isinstance(values, pl.Series):
            self.lf = self._df.with_columns(values)
        else:
            raise ValueError(
                f"Values must be a pandas Series, a polars Series, a list, or a numpy array. Got {type(values)}"
            )
