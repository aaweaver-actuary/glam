from __future__ import annotations
from typing import Protocol
import pandas as pd
import logging

logger = logging.getLogger(__name__)
file_handler = logging.FileHandler("preprocessor.log", mode="w")
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.setLevel(logging.INFO)


class BasePreprocessor(Protocol):
    """Protocol for preprocessing data."""

    @classmethod
    def new(self, df: pd.DataFrame) -> BasePreprocessor:
        """Create a new preprocessor."""
        ...

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the DataFrame."""
        ...


class NullPreprocessor:
    """Null implementation for preprocessing data.

    This class does not perform any preprocessing.
    """

    @classmethod
    def new(self, df: pd.DataFrame) -> BasePreprocessor:
        return NullPreprocessor()

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run all preprocessing steps."""
        return df


class DefaultPreprocessor:
    """Default implementation for preprocessing data.

    If a feature is categorical, replace missing values with the mode of the non-missing values.

    If a feature is numeric, replace missing values with the
    median of the non-missing values.
    """

    def __init__(self, df: pd.DataFrame):
        self._imputer = self._get_missing_value_imputer(df)

    @classmethod
    def new(self, df: pd.DataFrame) -> BasePreprocessor:
        return DefaultPreprocessor(df)

    @property
    def imputer(self) -> dict[str, str]:
        """Return the imputer for each column."""
        return self._imputer

    def _get_missing_value_imputer(self, df: pd.DataFrame) -> dict[str, str]:
        """Get the imputer for each column."""
        imputer = {}
        for col in df.columns:
            if df[col].dtype in ["object", "category"]:
                try:
                    imputer[col] = df[col].mode()[0]
                except IndexError:
                    logger.debug(
                        f"No mode found for column {col}. Just taking the first value."
                    )
                    imputer[col] = df[col][0]
            else:
                imputer[col] = df[col].median()
        return imputer

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the DataFrame."""
        for col in df.columns:
            df.loc[:, col] = df[col].fillna(self.imputer[col])
        return df

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run all preprocessing steps."""
        df = self.handle_missing_values(df)
        return df
