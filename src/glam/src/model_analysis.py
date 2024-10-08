"""An interface and implementation for a GLM model."""

from __future__ import annotations
from typing import Generator, Protocol

import logging
from glam.src.data import BaseModelData
from glam.src.fitters import BaseModelFitter
from glam.src.preprocessors import BasePreprocessor
from glam.src.data_splitter import BaseDataSplitter
from glam.src.models import BaseModelList

logger = logging.getLogger(__name__)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger.setLevel(logging.INFO)


class BaseModelAnalysis(Protocol):
    """Interface for a GLM model."""

    @property
    def data(self) -> BaseModelData:
        """Return the data object."""
        ...

    @property
    def fitter(self) -> BaseModelFitter:
        """Return the model fitter."""
        ...

    @property
    def splitter(self) -> BaseDataSplitter:
        """Return the data splitter."""
        ...

    @property
    def preprocessor(self) -> BasePreprocessor:
        """Return the preprocessor."""
        ...

    @property
    def models(self) -> BaseModelList:
        """Return the models."""
        ...

    @property
    def features(self) -> list[str]: ...  # noqa: D102

    @features.setter
    def features(self, feature: str) -> None: ...

    @property
    def interactions(self) -> list[str]: ...  # noqa: D102

    @interactions.setter
    def interactions(self, interactions: list[str]) -> None: ...

    def add_feature(self, feature: str) -> None: ...  # noqa: D102

    def add_interaction(self, *args: tuple[str]) -> None: ...  # noqa: D102

    def drop_feature(self, feature: str) -> None: ...  # noqa: D102

    def drop_features(self, features: list[str]) -> None: ...  # noqa: D102

    def fit_cv(self) -> Generator[BaseModelList, None, None]: ...  # noqa: D102
