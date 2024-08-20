from typing import Generator
import pandas as pd
import logging
import copy

from glam.src.data.base_model_data import BaseModelData
from glam.src.enums.model_task import ModelTask
from glam.src.fitters.base_model_fitter import BaseModelFitter
from glam.src.model_list.base_model_list import BaseModelList
from glam.src.fitted_model.base_fitted_model import BaseFittedModel
from glam.src.data.data_prep import BaseDataSplitter, BasePreprocessor
from glam.src.fitters.statsmodels_formula_glm_fitter import StatsmodelsFormulaGlmFitter
from glam.src.model_list.default_model_list import DefaultModelList
from glam.src.data.data_prep import TimeSeriesDataSplitter, DefaultPreprocessor
from glam.src.calculators.residual_calculators.binomial_glm_residual_calculator import (
    BinomialGlmResidualCalculator,
)

__all__ = ["BinomialGlmAnalysis"]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BinomialGlmAnalysis:
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
        self._fitter = fitter if fitter is not None else StatsmodelsFormulaGlmFitter()
        self._models = models if models is not None else DefaultModelList()
        self._fitted_model = fitted_model
        self._features = features if features is not None else []
        self._interactions = interactions if interactions is not None else []

        self._splitter = (
            splitter if splitter is not None else TimeSeriesDataSplitter(data)
        )
        self._preprocessor = (
            preprocessor if preprocessor is not None else DefaultPreprocessor(data)
        )

        self._task = task

    def __repr__(self):
        if len(self.features) > 0:
            return f"BinaryGlmAnalysis({self.linear_formula})"

        return f"BinaryGlmAnalysis({self.data.y.name} ~ 1)"

    def __str__(self):
        return self.__repr__()

    def copy(self):
        return copy.deepcopy(self)

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
        logger.debug(f"Adding interaction term: {args}")
        interaction_name = "___".join(args)
        if interaction_name in self.features:
            interaction_feature = self.data.df[interaction_name].copy()
        else:
            interaction_feature = self.data.df[args[0]].copy()
            for i, f in enumerate(args[1:]):
                # Ensure that the feature is available at all in the data
                if f not in self.data.feature_names:
                    logger.error(f"Feature {f} not in data. Continuing anyway...")
                    continue

                # Ensure that the feature is in the list of features
                # before adding it to an interaction term
                if f not in self.features:
                    logger.info(
                        f"Feature {f} not in current GLM feature list. Adding it now..."
                    )
                    self.add_feature(f)

                # Add the current feature to the interaction term
                interaction_feature *= self.data.df[f].copy()

        # Add the interaction feature to the data
        logger.debug(f"Added interaction feature: {interaction_feature.name}")
        logger.debug(f"Interaction feature: {self.data.df.columns.tolist()[-1]}")

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

    @property
    def feature_formula(self) -> str:
        """Return the formula for the model."""
        return " + ".join(self.features)

    @property
    def linear_formula(self) -> str:
        """Return the linear formula for the model."""
        return f"{self.data.y.name} ~ {self.feature_formula}"

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
        for X_train, y_train, X_test, y_test in self.X_y_generator:
            logger.debug(f"linear predictor: {self.linear_formula}")
            model = self.fitter.fit(self.linear_formula, X_train, y_train)
            self.models.add_model(model)
            yield self.models

    def fit(self) -> None:
        """Run the generator to fit the model for each cross-validation fold."""
        self.convert_data_to_floats()
        for _ in self.fit_cv():
            pass

    @property
    def resid(self) -> dict[str, pd.Series]:
        """Return the residuals of the model."""
        dat, names = [], []
        for i, model in enumerate(self.models.model_generator):
            dat.append(self.fitted_model.residuals(model))
            names.append(i + 1)

        return dict(zip(names, pd.Series(dat)))

    @property
    def mu(self) -> pd.Series:
        """Return the expected value of the response variable."""
        return self.models.model.mu

    def yhat(self, X: pd.DataFrame | None = None) -> pd.Series:
        """Return the predicted class."""
        if X is None:
            return self.models.model.mu.round(0)
        return self.models.model.predict(X).round(0)

    def yhat_prob(self, X: pd.DataFrame | None = None) -> pd.Series:
        """Return the predicted probability of the positive class."""
        if X is None:
            return self.mu
        return self.models.model.predict(X)

    def summary(self):
        """Return the summary of the model."""
        return self.models.model.summary()

    @property
    def residual_calculator(self) -> BinomialGlmResidualCalculator:
        if self.models.model is None:
            self.fit()

        coefficients = self.models.model.params
        return BinomialGlmResidualCalculator(
            self.X, self.y, self.yhat_prob(), coefficients
        )

    @property
    def deviance_residuals(self) -> pd.Series:
        return self.residual_calculator.deviance_residuals()

    @property
    def pearson_residuals(self) -> pd.Series:
        return self.residual_calculator.pearson_residuals(std=False)

    @property
    def std_pearson_residuals(self) -> pd.Series:
        return self.residual_calculator.pearson_residuals(std=True)

    def partial_residuals(self, feature: str) -> pd.Series:
        return self.residual_calculator.partial_residuals(feature)
