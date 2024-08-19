from __future__ import annotations
import pandas as pd
from typing import Protocol

__all__ = ["BasePreprocessor"]


class BasePreprocessor(Protocol):
    """Protocol for preprocessing data."""

    def __init__(self, df: pd.DataFrame) -> None: ...

    def new(self) -> BasePreprocessor:
        """Create a new preprocessor."""
        ...

    def run(self) -> pd.DataFrame:
        """Preprocess the DataFrame."""
        ...
