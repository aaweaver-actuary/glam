from concurrent.futures import ProcessPoolExecutor
import copy
from abc import ABC, abstractmethod
import pandas as pd
from typing import Generator

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
    def __init__(
        self,
        data: BaseModelData,
        fitter: BaseModelFitter | None = None,
        models: BaseModelList | None = None,
        features: list[str] | None = None,
        interactions: list[str] | None = None,
        fitted_model: BaseFittedModel | None = None,
        splitter: BaseDataSplitter | None = None,
        preprocessor: BasePreprocessor | None = None,
        task: ModelTask = ModelTask.CLASSIFICATION,
    ):
        self._data = data
        self._fitter = fitter
        self._models = models
        self._fitted_model = fitted_model
        self._features = features
        self._interactions = interactions
        self._splitter = splitter
        self._preprocessor = preprocessor
        self._task = task

    @abstractmethod
    def __repr__(self):
        pass

    def __str__(self):
        return self.__repr__()

    def copy(self):
        return copy.deepcopy(self)

    @classmethod
    def from_self(cls, self) -> "BaseAnalysis":
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
        return self._data

    @data.setter
    def data(self, data: BaseModelData) -> None:
        self._data = data

    @property
    def X_y_generator(self) -> tuple:
        for X_train, y_train, X_test, y_test in self.splitter.X_y_generator:
            yield X_train[self.features], y_train, X_test[self.features], y_test

    @property
    def fitter(self) -> BaseModelFitter:
        return self._fitter

    @property
    def splitter(self) -> BaseDataSplitter:
        return self._splitter

    @property
    def preproceessor(self) -> BasePreprocessor:
        return self._preprocessor

    @property
    def fitted_model(self) -> BaseFittedModel:
        return self._fitted_model

    @property
    def models(self) -> BaseModelList:
        return self._models

    @models.setter
    def models(self, models: BaseModelList) -> None:
        self._models = models

    def add_model(self, model) -> None:
        current_models = self.models
        current_models.add_model(model)
        self.models = current_models

    def get_model(self, index: int):
        return self.models.model_lookup[index]

    @property
    def features(self) -> list[str]:
        """Return the list of features used to fit the model."""
        try:
            return self._features
        except KeyError:
            try:
                output = []
                for g in self._features:
                    for f in self.data.feature_names:
                        if f.startswith(g):
                            output.append(f)
                return output

            except KeyError:
                raise KeyError

    @features.setter
    def features(self, features: list[str]) -> None:
        self._features = features

    def add_feature(self, feature: str) -> None:
        """Add the feature to the list of features."""
        cur_features = self.features
        is_not_already_included = feature not in cur_features
        is_available_in_data = feature in self.data.feature_names
        if is_not_already_included and is_available_in_data:
            self.features = cur_features + [feature]

    def add_interaction(self, *args: tuple[str]) -> None:
        """Add an interaction term by multiplying the features together."""
        interaction_name = "___".join(args)
        if interaction_name in self.features:
            interaction_feature = self.data.df[interaction_name].copy()
        else:
            interaction_feature = self.data.df[args[0]].copy()
            for i, f in enumerate(args[1:]):
                # Ensure that the feature is available at all in the data
                if f not in self.data.feature_names:
                    continue

                # Ensure that the feature is in the list of features
                # before adding it to an interaction term
                if f not in self.features:
                    self.add_feature(f)

                # Add the current feature to the interaction term
                interaction_feature *= self.data.df[f].copy()

        if interaction_name in self.data.feature_names:
            self.data.df[interaction_name] = interaction_feature
        else:
            interaction_feature.name = interaction_name
            self.data.add_feature(interaction_name, interaction_feature)

        self.add_feature(interaction_name)

    def drop_feature(self, feature: str) -> None:
        """Drop the feature from the list of features."""
        new_features = [f for f in self.features if f != feature]
        self.features = new_features

    def drop_features(self, *features) -> None:
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
        return self.data.cv.drop_duplicates().sort_values().to_list()[1:]

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
            if ((self.data.df[f].dtype == object) or (self.data.df[f].dtype == str))
        ]

    @property
    def numeric_features(self) -> list[str]:
        """Return the list of numeric features."""
        return [
            f
            for f in self.features
            if ((self.data.df[f].dtype == float) or (self.data.df[f].dtype == int))
        ]

    @abstractmethod
    def _fit_single_fold(
        self, X_train: pd.DataFrame, y_train: pd.Series
    ) -> BaseFittedModel:
        """Fits a single model for a cross-validation fold."""
        pass

    def convert_1_0_integers(self, x: pd.Series, offset: float = 1e-8) -> pd.DataFrame:
        """Convert the response variable to be 1s and 0s."""
        if x.dtype != float:
            n_unique = x.nunique()
            unique = x.unique()

            col_has_only_0s_and_1s = False
            if n_unique == 2:
                col_has_only_0s_and_1s = (unique[0] in [0, 1]) and (unique[1] in [0, 1])
            elif n_unique == 1:
                col_has_only_0s_and_1s = unique[0] in [0, 1]
            else:
                col_has_only_0s_and_1s = False

            if col_has_only_0s_and_1s:
                return x.replace({0: 0.0 + offset, 1: 1.0 - offset})
            else:
                return x
        else:
            return x

    def convert_data_to_floats(self) -> None:
        for col in self.features:
            self.data.df[col] = self.convert_1_0_integers(self.data.df[col])

    def fit_cv(self) -> Generator[BaseModelList, None, None]:
        """Fit/refit the model for each cross-validation fold using the current set of features."""
        for X_train, y_train, _, _ in self.X_y_generator:
            model = self._fit_single_fold(X_train, y_train)
            self.models.add_model(model)
            yield self.models

    def fit(self, parallel: bool = True) -> None:
        """Run the generator to fit the model for each cross-validation fold."""
        self.convert_data_to_floats()
        if parallel:
            with ProcessPoolExecutor() as executor:
                models = [
                    executor.submit(
                        self._fit_single_fold,
                        X,
                        y,
                    )
                    for X, y, _, _ in self.X_y_generator
                ]
                for model in models:
                    self.models.add_model(model.result())
        else:
            for _ in self.fit_cv():
                pass

    @property
    def mu(self) -> pd.Series:
        """Return the expected value of the response variable."""
        raise NotImplementedError

    @abstractmethod
    def yhat(self, X: pd.DataFrame | None = None) -> pd.Series:
        """Return the predicted class."""
        pass

    @abstractmethod
    def yhat_proba(self, X: pd.DataFrame | None = None) -> pd.Series:
        """Return the predicted probability of the positive class."""
        pass

    @property
    def residual_calculator(self) -> BaseResidualCalculator:
        raise NotImplementedError

    @abstractmethod
    def evaluate_new_feature(
        self, new_feature: str, parallel: bool = True
    ) -> pd.DataFrame:
        pass

    def _generate_new_feature_eval(
        self, new_features: list[str], parallel: bool = True
    ) -> Generator[pd.DataFrame, None, None]:
        for f in new_features:
            yield self.evaluate_new_feature(f, parallel).iloc[0]

    @abstractmethod
    def evaluate_new_features(self) -> pd.DataFrame:
        pass
