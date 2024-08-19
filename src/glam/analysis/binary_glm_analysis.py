from typing import Generator
import pandas as pd
import logging
import copy
from glam.src.data.base_model_data import BaseModelData
from glam.src.fitters.base_model_fitter import BaseModelFitter
from glam.src.model_list.base_model_list import BaseModelList
from glam.src.fitted_model.base_fitted_model import BaseFittedModel
from glam.src.data.data_prep import BaseDataSplitter, BasePreprocessor

from glam.src.fitters.statsmodels_formula_glm_fitter import StatsmodelsFormulaGlmFitter
from glam.src.model_list.default_model_list import DefaultModelList
from glam.src.fitted_model.statsmodels_fitted_glm import StatsmodelsFittedGlm
from glam.src.data.data_prep import TimeSeriesDataSplitter, DefaultPreprocessor

__all__ = ["BinaryGlmAnalysis"]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BinaryGlmAnalysis:
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
    ):
        self._data = data
        self._fitter = (
            fitter if fitter is not None else StatsmodelsFormulaGlmFitter(data)
        )
        self._models = models if models is not None else DefaultModelList()
        self._fitted_model = (
            fitted_model if fitted_model is not None else StatsmodelsFittedGlm(data)
        )

        self._features = features if features is not None else []
        self._interactions = interactions if interactions is not None else []

        self._splitter = (
            splitter if splitter is not None else TimeSeriesDataSplitter(data)
        )
        self._preprocessor = (
            preprocessor if preprocessor is not None else DefaultPreprocessor(data)
        )

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
        for X_train, y_train, X_test, y_test in self.splitter.X_y_generator(self.data):
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
        return self.models.model(index)

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

    @property
    def interactions(self) -> list[str]:
        """Return the list of interaction terms."""
        return self._interactions

    @interactions.setter
    def interactions(self, interactions: list[pd.Series]) -> None:
        """Add the interaction terms to the list of interactions."""
        self._interactions = interactions

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
        self.models = DefaultModelList()

        for X_train, y_train, X_test, y_test in self.X_y_generator:
            logger.debug(f"linear predictor: {self.linear_formula}")
            model = self.fitter.fit(X_train, y_train, self.linear_formula)
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
        return self.fitted_model.mu

    def yhat(self, X: pd.DataFrame | None = None) -> pd.Series:
        """Return the predicted class."""
        if X is None:
            X = self.X
        return self.fitted_model.yhat(X, self.get_model(9))

    def yhat_prob(self, X: pd.DataFrame | None = None) -> pd.Series:
        """Return the predicted probability of the positive class."""
        if X is None:
            X = self.X
        return self.fitted_model.yhat_prob(X, self.get_model(9))

    @property
    def accuracy(self) -> dict[int, float]:
        """Return the accuracy of the model for each cross-validation fold."""
        dat, names = [], []
        for i, model in enumerate(self.models.model_generator):
            X = self.X.loc[self.cv == i + 1]
            y = self.y[self.cv == i + 1]
            y_pred = self.fitted_model.yhat(X, model).round(0)
            dat.append((y_pred == y).mean())
            names.append(i + 1)

        return dict(zip(names, dat))

    @property
    def precision(self) -> dict[int, float]:
        """Return the precision of the model for each cross-validation fold."""
        dat, names = [], []
        for i, model in enumerate(self.models.model_generator):
            X = self.X.loc[self.cv == i + 1]
            y = self.y[self.cv == i + 1]
            y_pred = self.fitted_model.yhat(X, model).round(0)
            dat.append(((y_pred == 1) & (y == 1)).sum() / y_pred.sum())
            names.append(i + 1)

        return dict(zip(names, dat))

    @property
    def recall(self) -> dict[int, float]:
        """Return the recall of the model for each cross-validation fold."""
        dat, names = [], []
        for i, model in enumerate(self.models.model_generator):
            X = self.X.loc[self.cv == i + 1]
            y = self.y[self.cv == i + 1]
            y_pred = self.fitted_model.yhat(X, model).round(0)
            dat.append(((y_pred == 1) & (y == 1)).sum() / y.sum())
            names.append(i + 1)

        return dict(zip(names, dat))

    @property
    def f1(self) -> dict[int, float]:
        """Return the F1 score of the model for each cross-validation fold."""
        dat, names = [], []
        for i, model in enumerate(self.models.model_generator):
            X = self.X.loc[self.cv == i + 1]
            y = self.y[self.cv == i + 1]
            y_pred = self.fitted_model.yhat(X, model).round(0)
            precision = ((y_pred == 1) & (y == 1)).sum() / y_pred.sum()
            recall = ((y_pred == 1) & (y == 1)).sum() / y.sum()
            dat.append(2 * (precision * recall) / (precision + recall))
            names.append(i + 1)

        return dict(zip(names, dat))

    @property
    def auc_roc(self) -> dict[int, float]:
        """Return the AUC-ROC of the model for each cross-validation fold."""
        dat, names = [], []
        try:
            logger.debug("Calculating AUC-ROC")
            for i, model in enumerate(self.models.model_generator):
                X = self.X.loc[self.cv == i + 1]
                y = self.y[self.cv == i + 1]
                dat.append(self.fitted_model.roc_auc(X, y, model))
                names.append(i + 1)

            logger.debug("AUC-ROC calculation complete")
            return dict(zip(names, dat))
        except Exception as e:
            logger.error(f"Error in auc_roc property function: {e}")

    @property
    def ap(self) -> dict[int, float]:
        """Return the average precision of the model for each cross-validation fold."""
        dat, names = [], []
        try:
            logger.debug("Calculating average precision")
            for i, model in enumerate(self.models.model_generator):
                X = self.X.loc[self.cv == i + 1]
                y = self.y[self.cv == i + 1]
                dat.append(self.fitted_model.ap(X, y, model))
                names.append(i + 1)

            logger.debug("Average precision calculation complete")
            return dict(zip(names, dat))
        except Exception as e:
            logger.error(f"Error in ap property function: {e}")

    @property
    def log_loss(self) -> dict[int, float]:
        """Return the log loss of the model for each cross-validation fold."""
        dat, names = [], []
        try:
            logger.debug("Calculating log loss")
            for i, model in enumerate(self.models.model_generator):
                X = self.X.loc[self.cv == i + 1]
                y = self.y[self.cv == i + 1]
                dat.append(self.fitted_model.log_loss(X, y, model))
                names.append(i + 1)

            logger.debug("Log loss calculation complete")
            return dict(zip(names, dat))
        except Exception as e:
            logger.error(f"Error in log_loss property function: {e}")

    @property
    def breier(self) -> dict[int, float]:
        """Return the Brier score of the model for each cross-validation fold."""
        dat, names = [], []
        try:
            logger.debug("Calculating Breier score")
            for i, model in enumerate(self.models.model_generator):
                X = self.X.loc[self.cv == i + 1]
                y = self.y[self.cv == i + 1]
                dat.append(self.fitted_model.breier(X, y, model))
                names.append(i + 1)

            logger.debug("Breier score calculation complete")
            return dict(zip(names, dat))
        except Exception as e:
            logger.error(f"Error in breier property function: {e}")

    @property
    def aic(self) -> dict[int, float]:
        """Return the AIC of the model for each cross-validation fold."""
        dat, names = [], []
        try:
            logger.debug("Calculating AIC")
            for i, model in enumerate(self.models.model_generator):
                dat.append(self.fitted_model.aic(model))
                names.append(i + 1)

            logger.debug("AIC calculation complete")
            return dict(zip(names, dat))
        except Exception as e:
            logger.error(f"Error in aic property function: {e}")

    @property
    def bic(self) -> dict[int, float]:
        """Return the BIC of the model for each cross-validation fold."""
        dat, names = [], []
        try:
            logger.debug("Calculating BIC")
            for i, model in enumerate(self.models.model_generator):
                dat.append(self.fitted_model.bic(model))
                names.append(i + 1)

            logger.debug("BIC calculation complete")
            return dict(zip(names, dat))
        except Exception as e:
            logger.error(f"Error in bic property function: {e}")

    @property
    def deviance(self) -> dict[int, float]:
        """Return the deviance of the model for each cross-validation fold."""
        dat, names = [], []
        try:
            logger.debug("Calculating deviance")
            for i, model in enumerate(self.models.model_generator):
                dat.append(self.fitted_model.deviance(model))
                names.append(i + 1)

            logger.debug("Deviance calculation complete")
            return dict(zip(names, dat))
        except Exception as e:
            logger.error(f"Error in deviance property function: {e}")

    @property
    def mallow_cp(self) -> dict[int, float]:
        """Return the Mallows Cp of the model for each cross-validation fold."""
        dat, names = [], []
        try:
            logger.debug("Calculating Mallows Cp")
            for i, model in enumerate(self.models.model_generator):
                dat.append(self.fitted_model.mallow_cp(model))
                names.append(i + 1)

            logger.debug("Mallows Cp calculation complete")
            return dict(zip(names, dat))
        except Exception as e:
            logger.error(f"Error in mallow_cp property function: {e}")

    @property
    def performance(self):
        """Return the performance metrics for each cross-validation fold."""
        df = pd.DataFrame(
            {
                "accuracy": pd.Series(self.accuracy, name="accuracy"),
                "precision": pd.Series(self.precision, name="precision"),
                "recall": pd.Series(self.recall, name="recall"),
                "f1": pd.Series(self.f1, name="f1"),
                "auc_roc": pd.Series(self.auc_roc, name="auc_roc"),
                "aic": pd.Series(self.aic, name="aic"),
                "deviance": pd.Series(self.deviance, name="deviance"),
            }
        )

        df_mean = df.mean(axis=0)
        df_mean.name = "mean"

        df_std = df.std(axis=0)
        df_std.name = "std"

        df = pd.concat([df, df_mean.to_frame().T, df_std.to_frame().T], axis=0)

        return df.round(3)

    def summary(self):
        """Return the summary of the model."""
        try:
            return self.get_model(9).summary()
        except Exception as e:
            logger.error(f"Error in summary function: {e}")
