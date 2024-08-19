import pandas as pd
from glam.src.data.data_prep.preprocessors.base_preprocessor import BasePreprocessor

__all__ = ["NullPreprocessor"]

class NullPreprocessor:
    """Null implementation for preprocessing data.

    This class does not perform any preprocessing.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df

    @property
    def df(self) -> pd.DataFrame:
        """Return the DataFrame."""
        return self._df

    def new(self) -> BasePreprocessor:
        return NullPreprocessor(self.df)

    def run(self) -> pd.DataFrame:
        """Run all preprocessing steps."""
        return self.df
