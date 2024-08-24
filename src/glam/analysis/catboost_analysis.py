import pandas as pd
import logging
from concurrent.futures import ProcessPoolExecutor
from typing import Generator

from glam.analysis.base_analysis import BaseAnalysis
from glam.src.data.base_model_data import BaseModelData
from glam.src.enums.model_task import ModelTask
from glam.src.fitters.base_model_fitter import BaseModelFitter
from glam.src.model_list.base_model_list import BaseModelList
from glam.src.fitted_model.base_fitted_model import BaseFittedModel
from glam.src.data.data_prep import BaseDataSplitter, BasePreprocessor
from glam.src.fitters.catboost_fitter import CatboostFitter
from glam.src.model_list.default_model_list import DefaultModelList
from glam.src.data.data_prep import TimeSeriesDataSplitter, DefaultPreprocessor

__all__ = ["CatboostGbmAnalysis"]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CatboostGbmAnalysis(BaseAnalysis):
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
        super().__init__(
            data,
            fitter,
            models,
            features,
            interactions,
            fitted_model,
            splitter,
            preprocessor,
            task,
        )

        self._data = data
        self._fitter = fitter if fitter is not None else CatboostFitter()
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
            return f"CatboostGbmAnalysis({self.linear_formula})"

        return f"CatboostGbmAnalysis({self.data.y.name} ~ 1)"

    @property
    def mu(self) -> pd.Series:
        """Return the expected value of the response variable."""
        return pd.Series(self.models.model.mu, name="mu")

    def yhat(self, X: pd.DataFrame | None = None) -> pd.Series:
        """Return the predicted class."""
        if X is None:
            return self.models.model.mu.round(0)
        return self.models.model.predict(X).round(0)

    def yhat_proba(self, X: pd.DataFrame | None = None) -> pd.Series:
        """Return the predicted probability of the positive class."""
        if X is None:
            return self.mu
        return self.models.model.predict(X)

    @property
    def summary(self):
        """Return the summary of the model."""
        return self.models.model.summary()

    @property
    def coefficients(self) -> pd.Series:
        return self.models.model.coefficients

    @property
    def endog(self) -> pd.Series:
        return (
            pd.Series(self.models.model.model.data.y, name="endog").round(0).astype(int)
        )

    @property
    def exog(self) -> pd.DataFrame:
        return pd.DataFrame(
            self.models.model.model.data.X, columns=["Intercept"] + self.features
        )

    def _fit_single_fold(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> BaseFittedModel:
        return self.fitter.fit(X_train, y_train, X_test, y_test)

    def fit_cv(self) -> Generator[BaseModelList, None, None]:
        """Fit/refit the model for each cross-validation fold using the current set of features."""
        for X_train, y_train, X_test, y_test in self.X_y_generator:
            model = self._fit_single_fold(X_train, y_train, X_test, y_test)
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
                        X_train,
                        y_train,
                        X_test,
                        y_test,
                    )
                    for X_train, y_train, X_test, y_test in self.X_y_generator
                ]
                for model in models:
                    self.models.add_model(model.result())
        else:
            for _ in self.fit_cv():
                pass

    def evaluate_new_feature(
        self, new_feature: str, parallel: bool = True
    ) -> pd.DataFrame:
        pass

    def evaluate_new_features(self) -> pd.DataFrame:
        pass
