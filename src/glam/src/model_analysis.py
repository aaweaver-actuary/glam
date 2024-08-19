"""This module provides an interface and implementation for a GLM model."""

import copy
from typing import Generator, Protocol

import numpy as np
import pandas as pd

# from hit_ratio.old_glm.model_data import BaseModelData
from glam.src.data import BaseModelData
from hit_ratio.old_glm.model_fitter import (
    BaseModelFitter,
    StatsmodelsFormulaGlmFitter,
)
from hit_ratio.old_glm.model_list import BaseModelList, DefaultModelList
from hit_ratio.old_glm.model_result import BaseModelResult, StatsmodelsGlmResult
from hit_ratio.old_glm.splitter import BaseDataSplitter, TimeSeriesDataSplitter
from hit_ratio.old_glm.preprocessor import BasePreprocessor, DefaultPreprocessor

import logging

logger = logging.getLogger(__name__)
file_handler = logging.FileHandler("model_analysis__package.log", mode="w")
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.setLevel(logging.INFO)


class BaseModelAnalysis(Protocol):
    @property
    def data(self) -> BaseModelData: ...

    @property
    def fitter(self) -> BaseModelFitter: ...

    @property
    def splitter(self) -> BaseDataSplitter: ...

    @property
    def preprocessor(self) -> BasePreprocessor: ...

    @property
    def models(self) -> BaseModelList: ...

    @property
    def features(self) -> list[str]: ...

    @features.setter
    def features(self, feature: str) -> None: ...

    @property
    def interactions(self) -> list[str]: ...

    @interactions.setter
    def interactions(self, interactions: list[str]) -> None: ...

    def add_feature(self, feature: str) -> None: ...

    def add_interaction(self, *args: tuple[str]) -> None: ...

    def drop_feature(self, feature: str) -> None: ...

    def drop_features(self, features: list[str]) -> None: ...

    def fit_cv(self) -> Generator[BaseModelList, None, None]: ...


