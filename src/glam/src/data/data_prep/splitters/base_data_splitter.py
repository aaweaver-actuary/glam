import pandas as pd
from typing import Protocol, Generator

from glam.src.data.base_model_data import BaseModelData

__all__ = ["BaseDataSplitter"]


class BaseDataSplitter(Protocol):
    """Protocol for splitting data."""

    def __init__(self, data: BaseModelData) -> None: ...

    def split_data(
        self, fold: int
    ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """Split the data into training and testing sets for the given fold."""
        ...

    @property
    def X_y_generator(
        self,
    ) -> Generator[tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series], None, None]:
        """Generate train-test splits for cross-validation."""
        ...

    @property
    def fold_labels(self) -> list[int]:
        """Return the list of unique fold labels."""
        ...
