from concurrent.futures import ProcessPoolExecutor
from abc import abstractmethod
import pandas as pd

from glam.src.data.base_model_data import BaseModelData
from glam.src.fitters.base_model_fitter import BaseModelFitter
from glam.src.data.data_prep import BaseDataSplitter, BasePreprocessor
from glam.src.model_list.base_model_list import BaseModelList
from glam.src.fitted_model.base_fitted_model import BaseFittedModel
from glam.src.enums.model_task import ModelTask
from glam.src.calculators.residual_calculators.base_residual_calculator import (
    BaseResidualCalculator,
)
from glam.analysis.base_analysis import BaseAnalysis

__all__ = ["BaseGlmAnalysis"]


class BaseGlmAnalysis(BaseAnalysis):
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

    @abstractmethod
    def __repr__(self):
        pass

    @property
    def feature_formula(self) -> str:
        """Return the formula for the model."""
        return " + ".join(self.features)

    @property
    def linear_formula(self) -> str:
        """Return the linear formula for the model."""
        return f"{self.data.y.name} ~ {self.feature_formula}"

    def _fit_single_fold(
        self, X_train: pd.DataFrame, y_train: pd.Series
    ) -> BaseFittedModel:
        """Fits a single model for a cross-validation fold."""
        return self.fitter.fit(self.linear_formula, X_train, y_train)

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
    def summary(self):
        """Return the summary of the model."""
        raise NotImplementedError

    @property
    def coefficients(self) -> pd.Series:
        raise NotImplementedError

    @property
    def endog(self) -> pd.Series:
        raise NotImplementedError

    @property
    def exog(self) -> pd.DataFrame:
        raise NotImplementedError

    @property
    def residual_calculator(self) -> BaseResidualCalculator:
        raise NotImplementedError

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

    @property
    def aic(self) -> float:
        raise NotImplementedError

    @property
    def bic(self) -> float:
        raise NotImplementedError

    @property
    def deviance(self) -> float:
        raise NotImplementedError

    @property
    def leverage(self) -> pd.Series:
        raise NotImplementedError

    @abstractmethod
    def evaluate_new_feature(
        self, new_feature: str, parallel: bool = True
    ) -> pd.DataFrame:
        pass

    def evaluate_new_features(
        self,
        new_features: list[str] | None = None,
        parallel: bool = True,
        p_value_cutoff: float = 0.05,
    ) -> pd.DataFrame:
        if new_features is None:
            new_features = self.remaining_features

        if parallel:
            with ProcessPoolExecutor() as executor:
                futures = [
                    executor.submit(self.evaluate_new_feature, f, parallel)
                    for f in new_features
                ]

                # pre-assign arrays to hold the results
                idx = [0] * len(futures)
                deviance = [0.0] * len(futures)
                aic = [0.0] * len(futures)
                bic = [0.0] * len(futures)
                dof = [0] * len(futures)
                p_value = [0.0] * len(futures)

                for i, f in enumerate(futures):
                    idx[i] = i
                    deviance[i], aic[i], bic[i], dof[i], p_value[i] = f.result().iloc[0]

                n_to_remove = (
                    pd.Series(p_value).astype(float).ge(float(p_value_cutoff)).sum()
                )
                if n_to_remove > 0:
                    print(
                        f"Feature evaluation has removed {n_to_remove} features with p-values greater than {p_value_cutoff:.1%}."
                    )
                output = (
                    pd.DataFrame(
                        {
                            "Model": [f"[Current Model] + {f}" for f in new_features],
                            "Deviance": deviance,
                            "AIC": aic,
                            "BIC": bic,
                            "DofF": dof,
                            "p_value": p_value,
                        }
                    )
                    .sort_values(by="BIC", ascending=True)
                    .set_index("Model")
                )
                return output.loc[
                    output["p_value"].astype(float) < float(p_value_cutoff)
                ]
        else:
            return pd.concat(
                [df for df in self._generate_new_feature_eval(new_features, parallel)],
                axis=1,
            ).T
