import pandas as pd
from glam.src.fitted_model.base_fitted_model import BaseFittedModel
from glam.src.enums import ModelTask


class CatboostFitter:
    """Implement the BaseModelFitter protocol to fit a Catboost gradient-boosted tree model.

    Attributes
    ----------
    fitted_model : BaseFittedModel | None
        The fitted model. If None, the model has not been fitted yet.
    task : ModelTask
        The task of the model (classification or regression). Uses the ModelTask enum.
    **kwargs
        Additional keyword arguments to pass to the Catboost model.

    Methods
    -------
    **fit(formula: str, X: pd.DataFrame, y: pd.Series) -> catboost.CatBoostClassifier | catboost.CatBoostRegressor**

        Fit either a CatBoostClassifier or CatBoostRegressor model using the formula and data.
    """

    def __init__(
        self,
        fitted_model: BaseFittedModel | None = None,
        task: ModelTask = ModelTask.CLASSIFICATION,
        **kwargs,
    ):
        self._fitted_model = fitted_model
        self._task = task
        self._kwargs = kwargs

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(task={self.task})"

    def __str__(self) -> str:
        return self.__repr__()

    @property
    def fitted_model(self):
        return self._fitted_model

    @fitted_model.setter
    def fitted_model(self, fitted_model: BaseFittedModel):
        self._fitted_model = fitted_model

    @property
    def task(self):
        return self._task

    def _fit_classifier(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> BaseFittedModel:
        from catboost import CatBoostClassifier

        model = CatBoostClassifier(verbose=False, **self._kwargs)
        model.fit(X_train, y_train, eval_set=(X_test, y_test))

        return model

    def _fit_regressor(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> BaseFittedModel:
        from catboost import CatBoostRegressor

        model = CatBoostRegressor(verbose=False, **self._kwargs)
        model.fit(X_train, y_train, eval_set=(X_test, y_test))

        return model

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> BaseFittedModel:
        fitted_model = (
            self._fit_classifier(X_train, y_train, X_test, y_test)
            if self.task == ModelTask.CLASSIFICATION
            else self._fit_regressor(X_train, y_train, X_test, y_test)
        )

        self.fitted_model = fitted_model
        return fitted_model
