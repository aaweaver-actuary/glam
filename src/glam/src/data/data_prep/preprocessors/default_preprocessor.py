import pandas as pd

from glam.src.data.data_prep.preprocessors.base_preprocessor import BasePreprocessor
from glam.src.data.base_model_data import BaseModelData

__all__ = ["DefaultPreprocessor"]


class DefaultPreprocessor:
    """Default implementation for preprocessing data.

    If a feature is categorical, replace missing values with the mode of the non-missing values.

    If a feature is numeric, replace missing values with the
    median of the non-missing values.
    """

    def __init__(self, data: BaseModelData) -> None:
        self._imputer = self._get_missing_value_imputer(data.df)
        self._data = data

    @property
    def data(self) -> BaseModelData:
        """Return the BaseModelData."""
        return self._data

    @data.setter
    def data(self, new_data: BaseModelData) -> None:
        self._data = new_data

    @property
    def df(self) -> pd.DataFrame:
        """Return the DataFrame."""
        return self.data.df

    @property
    def new(self) -> BasePreprocessor:
        return DefaultPreprocessor(self.data)

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
                    imputer[col] = df[col].dropna().mode()[0]
                except IndexError:
                    imputer[col] = df[col].dropna()[0]
                except KeyError:
                    imputer[col] = df[col].dropna()[0]
            else:
                imputer[col] = df[col].median()
        return imputer

    def handle_missing_values(self) -> None:
        """Handle missing values in the DataFrame."""
        df = self.df
        for col in self.df.columns:
            df.loc[:, col] = self.df[col].fillna(self.imputer[col])
        self.data.df = df

    def run(self) -> pd.DataFrame:
        """Run all preprocessing steps."""
        self.handle_missing_values()
        return self.df
