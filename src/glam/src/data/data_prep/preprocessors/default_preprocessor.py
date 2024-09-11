"""Module providing the DefaultPreprocessor class for preprocessing data."""

from __future__ import annotations
import pandas as pd

from glam.src.data.data_prep.preprocessors.base_preprocessor import BasePreprocessor
from glam.src.data.base_model_data import BaseModelData

__all__ = ["DefaultPreprocessor"]


class DefaultPreprocessor:
    """Default implementation for preprocessing data.

    If a feature is categorical, replace missing values with the mode of the non-missing values.

    If a feature is numeric, replace missing values with the
    median of the non-missing values.

    Attributes
    ----------
    data : BaseModelData
        The BaseModelData object containing the data to be preprocessed.
    imputer : dict[str, str]
        A dictionary containing the imputer for each column.

    Parameters
    ----------
    data : BaseModelData
        The BaseModelData object containing the data to be preprocessed.

    Methods
    -------
    handle_missing_values()
        Handle missing values in the DataFrame. Replaces missing values with the imputer value.
    run()
        Run all preprocessing steps.
    """

    def __init__(self, data: BaseModelData) -> None:
        self._imputer = self._get_missing_value_imputer(data.df)
        self._data = data

    @property
    def data(self) -> BaseModelData:
        """Return the BaseModelData object."""
        return self._data

    @data.setter
    def data(self, new_data: BaseModelData) -> None:
        """Set the BaseModelData object.

        Parameters
        ----------
        new_data : BaseModelData
            The new BaseModelData object.
        """
        self._data = new_data
        self._imputer = self._get_missing_value_imputer(new_data.df)

    @property
    def df(self) -> pd.DataFrame:
        """Return the DataFrame."""
        return self.data.df

    @property
    def new(self) -> BasePreprocessor:
        """Create a new instance of the DefaultPreprocessor class.

        Returns
        -------
        BasePreprocessor
            A new instance of the DefaultPreprocessor class.
        """
        return DefaultPreprocessor(self.data)

    @property
    def imputer(self) -> dict[str, str]:
        """Return the imputer for each column."""
        return self._imputer

    def _get_missing_value_imputer(self, df: pd.DataFrame) -> dict[str, str]:
        """Get the imputer for each column.

        - If a feature is categorical, replace missing values with the mode of the non-missing values.
        - If a feature is numeric, replace missing values with the median of the non-missing values.

        Special cases
        -------------
        - If the mode is not available, use the first non-missing value.
        - If the first non-missing value is not available, use the first value.
        - If the first value is not available, use the first value after forward filling.
        - If forward filling is not available, use the first value after backward filling.
        - If backward filling is not available, use the first value after forward filling and backward filling.
        - If all the above cases fail, use the first value after forward filling and backward filling.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame containing the data.

        Returns
        -------
        dict[str, str]
            A dictionary containing the imputer for each column. The key is the column name, and the
            value is the imputer value. Elsewhere, this is referred to as "the imputer for each column".
        """
        imputer = {}
        for col in df.columns:
            if df[col].dtype in ["object", "category"]:
                # If all values are missing, use None
                if df[col].dropna().shape[0] == 0:
                    imputer[col] = None
                    continue
                try:
                    imputer[col] = df[col].dropna().mode()[0]
                except IndexError:
                    imputer[col] = df[col].dropna()[0]
                except KeyError:
                    if df[col].dropna().shape[0] == 0:
                        imputer[col] = ""
                    else:
                        imputer[col] = df[col].fillna("ffill").iloc[0]
                except Exception:
                    try:
                        imputer[col] = df[col].ffill().bfill().mode()[0]
                    except Exception:
                        imputer[col] = df[col].ffill().bfill().iloc[0]

            else:
                imputer[col] = df[col].median()
        return imputer

    def handle_missing_values(self) -> None:
        """Handle missing values in the DataFrame.

        Replaces missing values with the imputer value.
        """
        df = self.df

        # Don't do anything if there are no missing values
        if df.isna().sum().sum() == 0:
            return

        # Don't do anything if there are only missing values (no way to impute)
        if df.isna().all().all():
            return

        # Otherwise, impute missing values with the dictionary of imputer values
        for col in self.df.columns:
            df.loc[:, col] = self.df[col].fillna(self.imputer[col])
        self.data.df = df

    def run(self) -> pd.DataFrame:
        """Run all preprocessing steps."""
        self.handle_missing_values()
        return self.df
