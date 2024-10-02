"""Define the base class for analyses."""

from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor
import copy
from abc import ABC, abstractmethod
import pandas as pd
from typing import Generator, Optional

from glam.src.data.base_model_data import BaseModelData
from glam.src.fitters.base_model_fitter import BaseModelFitter
from glam.src.data.data_prep import BaseDataSplitter, BasePreprocessor
from glam.src.model_list.base_model_list import BaseModelList
from glam.src.fitted_model.base_fitted_model import BaseFittedModel
from glam.src.enums.model_task import ModelTask
from glam.src.calculators.residual_calculators.base_residual_calculator import (
    BaseResidualCalculator,
)

__all__ = ["BaseAnalysis"]


class BaseAnalysis(ABC):
    """Define the base class for analyses."""

    def __init__(
        self,
        data: BaseModelData,
        fitter: Optional[BaseModelFitter] = None,
        models: Optional[BaseModelList] = None,
        features: Optional[list[str]] = None,
        interactions: Optional[list[str]] = None,
        fitted_model: Optional[BaseFittedModel] = None,
        splitter: Optional[BaseDataSplitter] = None,
        preprocessor: Optional[BasePreprocessor] = None,
        task: ModelTask = ModelTask.CLASSIFICATION,
    ):
        self._data: BaseModelData = data
        self._fitter: Optional[BaseModelFitter] = fitter
        self._models: Optional[BaseModelList] = models
        self._fitted_model: Optional[BaseFittedModel] = fitted_model
        self._features: Optional[list[str]] = features
        self._interactions: Optional[list[str]] = interactions
        self._splitter: Optional[BaseDataSplitter] = splitter
        self._preprocessor: Optional[BasePreprocessor] = preprocessor
        self._task: ModelTask = task

    @abstractmethod
    def __repr__(self) -> str:
        """Return the string representation of the class."""

    def __str__(self) -> str:
        """Return the string representation of the class."""
        return self.__repr__()

    def copy(self) -> BaseAnalysis:
        """Return a deep copy of the analysis."""
        return copy.deepcopy(self)

    @classmethod
    def from_self(cls, self: BaseAnalysis) -> BaseAnalysis:
        """Create a new instance of the class from an existing instance."""
        return cls(
            data=self._data,
            fitter=self._fitter,
            models=self._models,
            features=self._features,
            interactions=self._interactions,
            fitted_model=self._fitted_model,
            splitter=self._splitter,
            preprocessor=self._preprocessor,
            task=self._task,
        )

    @property
    def data(self) -> BaseModelData:
        """Return the ModelData object containing the data used to fit the model."""
        return self._data

    @data.setter
    def data(self, data: BaseModelData) -> None:
        """Update the data object."""
        self._data = data

    @property
    def X_y_generator(
        self,
    ) -> Generator[tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series], None, None]:
        """Return a generator that yields the training and testing data for each cross-validation fold."""
        if self._splitter is None:
            raise ValueError("Splitter is not defined.")
        for X_train, y_train, X_test, y_test in self._splitter.X_y_generator:
            yield X_train[self.features], y_train, X_test[self.features], y_test

    @property
    def fitter(self) -> Optional[BaseModelFitter]:
        """Return the fitter object."""
        return self._fitter

    @property
    def splitter(self) -> Optional[BaseDataSplitter]:
        """Return the splitter object."""
        return self._splitter

    @property
    def preprocessor(self) -> Optional[BasePreprocessor]:
        """Return the preprocessor object."""
        return self._preprocessor

    @property
    def fitted_model(self) -> Optional[BaseFittedModel]:
        """Return the fitted model."""
        return self._fitted_model

    @property
    def models(self) -> Optional[BaseModelList]:
        """Return the list of fitted models."""
        return self._models

    @models.setter
    def models(self, models: BaseModelList) -> None:
        """Update the list of fitted models."""
        self._models = models

    def add_model(self, model: BaseFittedModel) -> None:
        """Add the model to the list of fitted models."""
        if self._models is None:
            raise ValueError("Model list is not defined.")
        self._models.add_model(model)

    def get_model(self, index: int) -> BaseFittedModel:
        """Get a specific model from the list of fitted models."""
        if self._models is None:
            raise ValueError("Model list is not defined.")
        return self._models.model_lookup[index]

    @property
    def features(self) -> list[str]:
        """Return the list of features used to fit the model."""
        if self._features is None:
            raise ValueError("Features are not defined.")
        return self._features

    @features.setter
    def features(self, features: list[str]) -> None:
        self._features = features

    def add_feature(self, feature: str) -> None:
        """Add the feature to the list of features."""
        if feature not in self.features and feature in self.data.feature_names:
            self.features = [*self.features, feature]

    def add_interaction(self, *args: str) -> None:
        """Add an interaction term by multiplying the features together."""
        interaction_name = "___".join(args)
        if interaction_name not in self.features:
            interaction_feature = self.data.df[args[0]].copy()
            for f in args[1:]:
                if f not in self.data.feature_names:
                    raise ValueError(f"Feature {f} not found in data.")
                interaction_feature *= self.data.df[f]
            self.data.add_feature(interaction_name, interaction_feature)
            self.add_feature(interaction_name)

    def drop_feature(self, feature: str) -> None:
        """Drop the feature from the list of features."""
        self.features = [f for f in self.features if f != feature]

    def drop_features(self, *features: str) -> None:
        """Drop the features from the list of features."""
        for f in features:
            self.drop_feature(f)

    @property
    def X(self) -> pd.DataFrame:
        """Return the feature matrix."""
        return self.data.X[self.features]

    @property
    def y(self) -> pd.Series:
        """Return the response variable."""
        return self.data.y

    @property
    def cv(self) -> pd.Series:
        """Return the cross-validation fold."""
        return self.data.cv

    @property
    def cv_list(self) -> list[int]:
        """Return the list of unique cross-validation folds, excluding the first."""
        return self.data.cv.drop_duplicates().sort_values().tolist()[1:]

    @property
    def unanalyzed(self) -> list[str]:
        """Return the list of unanalyzed features."""
        return self.data.unanalyzed

    @property
    def remaining_features(self) -> list[str]:
        """Return the list of features that have not been analyzed."""
        return [f for f in self.data.feature_names if f not in self.features]

    @property
    def string_features(self) -> list[str]:
        """Return the list of string features."""
        return [
            f
            for f in self.features
            if self.data.df[f].dtype == object or self.data.df[f].dtype == "str"
        ]

    @property
    def numeric_features(self) -> list[str]:
        """Return the list of numeric features."""
        return [f for f in self.features if self.data.df[f].dtype in ["float", "int"]]

    @abstractmethod
    def _fit_single_fold(
        self, X_train: pd.DataFrame, y_train: pd.Series
    ) -> BaseFittedModel:
        """Fits a single model for a cross-validation fold."""

    def convert_1_0_integers(self, x: pd.Series, offset: float = 1e-8) -> pd.Series:
        """Convert the response variable to be 1s and 0s."""
        if x.dtype != float:
            unique_values = x.unique()
            if set(unique_values).issubset({0, 1}):
                return x.replace({0: 0.0 + offset, 1: 1.0 - offset})
        return x

    def convert_data_to_floats(self) -> None:
        """Convert the data to floats."""
        for col in self.features:
            self.data.df[col] = self.convert_1_0_integers(self.data.df[col])

    def fit_cv(self) -> Generator[BaseModelList, None, None]:
        """Fit/refit the model for each cross-validation fold using the current set of features."""
        for X_train, y_train, _, _ in self.X_y_generator:
            model = self._fit_single_fold(X_train, y_train)
            if self._models is None:
                raise ValueError("Model list is not defined.")
            self._models.add_model(model)
            yield self._models

    def fit(self, parallel: bool = False) -> None:
        """Run the generator to fit the model for each cross-validation fold."""
        self.convert_data_to_floats()
        if parallel:
            with ProcessPoolExecutor() as executor:
                futures = [
                    executor.submit(self._fit_single_fold, X, y)
                    for X, y, _, _ in self.X_y_generator
                ]
                for future in futures:
                    if self._models is None:
                        raise ValueError("Model list is not defined.")
                    self._models.add_model(future.result())
        else:
            for _ in self.fit_cv():
                pass

    @property
    def mu(self) -> pd.Series:
        """Return the expected value of the response variable."""
        raise NotImplementedError

    @abstractmethod
    def yhat(self, X: Optional[pd.DataFrame] = None) -> pd.Series:
        """Return the predicted class."""

    @abstractmethod
    def yhat_proba(self, X: Optional[pd.DataFrame] = None) -> pd.Series:
        """Return the predicted probability of the positive class."""

    @property
    def residual_calculator(self) -> BaseResidualCalculator:
        """Return the residual calculator object."""
        raise NotImplementedError

    @abstractmethod
    def evaluate_new_feature(
        self, new_feature: str, parallel: bool = False
    ) -> pd.DataFrame:
        """Return an evaluation of a potential new feature."""

    def _generate_new_feature_eval(
        self, new_features: list[str], parallel: bool = False
    ) -> Generator[pd.Series, None, None]:
        for f in new_features:
            yield self.evaluate_new_feature(f, parallel).iloc[0]

    @abstractmethod
    def evaluate_new_features(self) -> pd.DataFrame:
        """Evaluate the potential new features."""
