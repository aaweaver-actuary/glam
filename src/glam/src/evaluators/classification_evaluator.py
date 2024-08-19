import numpy as np
import statsmodels
from sklearn.metrics import roc_auc_score, roc_curve

__all__ = ["ClassificationEvaluator"]


class ClassificationEvaluator:
    """Implementation for evaluating classification models."""

    def tpr(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model: statsmodels.genmod.generalized_linear_model.GLMResults,
    ) -> np.ndarray:
        """Return the true positive rate."""
        y_pred = model.predict(X)
        y_true = y
        return np.sum((y_pred == 1) & (y_true == 1)) / np.sum(y_true == 1)

    def fpr(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model: statsmodels.genmod.generalized_linear_model.GLMResults,
    ) -> np.ndarray:
        """Return the false positive rate."""
        y_pred = model.predict(X)
        y_true = y
        return np.sum((y_pred == 1) & (y_true == 0)) / np.sum(y_true == 0)

    def accuracy(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model: statsmodels.genmod.generalized_linear_model.GLMResults,
    ) -> np.ndarray:
        """Return the accuracy of the model."""
        y_pred = model.predict(X)
        y_true = y
        return np.sum(y_pred == y_true) / len(y_true)

    def precision(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model: statsmodels.genmod.generalized_linear_model.GLMResults,
    ) -> np.ndarray:
        """Return the precision of the model."""
        y_pred = model.predict(X)
        y_true = y
        return np.sum((y_pred == 1) & (y_true == 1)) / np.sum(y_pred == 1)

    def recall(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model: statsmodels.genmod.generalized_linear_model.GLMResults,
    ) -> np.ndarray:
        """Return the recall of the model."""
        y_pred = model.predict(X)
        y_true = y
        return np.sum((y_pred == 1) & (y_true == 1)) / np.sum(y_true == 1)

    def f1_score(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model: statsmodels.genmod.generalized_linear_model.GLMResults,
    ) -> np.ndarray:
        """Return the F1 score of the model."""
        precision = self.precision(X, y, model)
        recall = self.recall(X, y, model)
        return 2 * (precision * recall) / (precision + recall)

    def roc_auc(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model: statsmodels.genmod.generalized_linear_model.GLMResults,
    ) -> np.ndarray:
        """Return the ROC AUC of the model."""
        y_pred = model.predict(X)
        return roc_auc_score(y, y_pred)

    def roc_curve(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model: statsmodels.genmod.generalized_linear_model.GLMResults,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return the ROC curve of the model."""
        y_pred = model.predict(X)
        return roc_curve(y, y_pred)

    def aic(
        self, model: statsmodels.genmod.generalized_linear_model.GLMResults
    ) -> float:
        """Return the AIC of the model."""
        return model.aic

    def bic(
        self, model: statsmodels.genmod.generalized_linear_model.GLMResults
    ) -> float:
        """Return the BIC of the model."""
        return model.bic

    def deviance(
        self, model: statsmodels.genmod.generalized_linear_model.GLMResults
    ) -> float:
        """Return the deviance of the model."""
        return model.deviance

    def evaluate(self) -> None:
        """Evaluate the classification models."""
