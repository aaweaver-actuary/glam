import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from glam.src.fitted_model.base_fitted_model import BaseFittedModel
from glam.src.data.base_model_data import BaseModelData

__all__ = ["ClassificationEvaluator"]


class ClassificationEvaluator:
    """Concrete implementation of the classification model evaluator.

    Attributes
    ----------
    data : BaseModelData
        The ModelData object containing the data used to fit the model.
    model : BaseFittedModel
        The fitted model object.
    X : pd.DataFrame
        The features used to fit the model.
    y : pd.Series
        The response variable used to fit the model.
    yhat_proba : pd.Series
        The predicted response variable. For a binary classification model, this is the probability of the positive class.
    yhat : pd.Series
        The predicted response variable. For a binary classification model, this is the predicted class.
    n_actually_positive : int
        The number of positive cases in the response variable.
    n_actually_negative : int
        The number of negative cases in the response variable.
    n_positive_predictions : int
        The number of positive predictions.
    n_correct_predictions : int
        The number of correct predictions.
    n_true_positives : int
        The number of true positive predictions.
    n_false_positives : int
        The number of false positive predictions.
    tpr : float
        The true positive rate.
    fpr : float
        The false positive rate.
    accuracy : float
        The accuracy of the model.
    precision : float
        The precision of the model.
    recall : float
        The recall of the model.
    f1_score : float
        The F1 score of the model.
    roc_auc : float
        The ROC AUC of the model.
    roc_curve : tuple
        The ROC curve of the model.
    aic : float
        The AIC of the model.
    deviance : float
        The deviance of the model.

    Methods
    -------
    **__init__(data: BaseModelData, model: BaseFittedModel) -> None**

        Initialize the object with the given data and model.
    **__repr__() -> str**

        Return a string representation of the object.
    **__str__() -> str**

        Return a string representation of the object.
    **evaluate() -> None**

        Evaluate the classification models.
    """

    def __init__(self, data: BaseModelData, model: BaseFittedModel) -> None:
        self._data = data
        self._model = model

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model})"

    def __str__(self) -> str:
        return self.__repr__()

    @property
    def data(self) -> BaseModelData:
        """Return the ModelData object containing the data used to fit the model."""
        return self._data

    @property
    def model(self) -> BaseFittedModel:
        """Return the fitted model object."""
        return self._model

    @property
    def X(self) -> pd.DataFrame:
        """Return the feature matrix used to fit the model."""
        return self.data.X

    @property
    def y(self) -> pd.Series:
        """Return the response variable used to fit the model."""
        return self.data.y

    @property
    def yhat_proba(self) -> pd.Series:
        """Return the predicted probability of the positive class."""
        return self.model.mu

    @property
    def yhat(self) -> pd.Series:
        """Return the predicted class."""
        return self.yhat_proba.round(0)

    @property
    def n_actually_positive(self) -> int:
        """Return the number of true actuals."""
        return self.y.eq(1).sum()

    @property
    def n_actually_negative(self) -> int:
        """Return the number of true negatives."""
        return self.y.eq(0).sum()

    @property
    def n_positive_predictions(self) -> int:
        """Return the number of positive predictions."""
        return self.yhat.eq(1).sum()

    @property
    def n_correct_predictions(self) -> int:
        """Return the number of true predictions."""
        return (self.yhat == self.y).sum()

    @property
    def n_true_positives(self) -> int:
        """Return the number of true positive predictions."""
        return (self.yhat.eq(1) & self.y.eq(1)).sum()

    @property
    def n_false_positives(self) -> int:
        """Return the number of false positive predictions."""
        return (self.yhat.eq(1) & self.y.eq(0)).sum()

    @property
    def tpr(self) -> float:
        """Return the true positive rate."""
        return self.n_true_positives() / self.n_actually_positive()

    @property
    def fpr(self) -> float:
        """Return the false positive rate."""
        return self.n_false_positives() / self.n_actually_negative()

    @property
    def accuracy(self) -> float:
        """Return the accuracy of the model."""
        return self.n_correct_predictions() / self.y.shape[0]

    @property
    def precision(self) -> float:
        """Return the precision of the model.

        Precision is an appropriate metric when the cost of false positives (eg false alarms) is high.
        """
        return self.n_true_positives() / self.n_positive_predictions()

    @property
    def recall(self) -> float:
        """Return the recall of the model.

        Recall is an appropriate metric when the cost of false negatives (eg missing a positive) is high.
        """
        return self.tpr()

    @property
    def f1_score(self) -> float:
        """Return the F1 score of the model.

        The F1 score is the harmonic mean of precision and recall. It is an appropriate metric
        when the cost of false positives and false negatives are both high.
        """
        return 2 * (self.precision * self.recall) / (self.precision + self.recall)

    @property
    def roc_auc(self) -> float:
        """Return the ROC AUC of the model."""
        return roc_auc_score(self.y, self.yhat_proba)

    @property
    def roc_curve(self) -> tuple:
        """Return the ROC curve of the model."""
        return roc_curve(self.y, self.yhat_proba)

    @property
    def aic(self) -> float:
        """Return the AIC of the model."""
        return self.model.aic

    @property
    def deviance(self) -> float:
        """Return the deviance of the model."""
        return self.model.deviance

    def evaluate(self) -> None:
        """Evaluate the classification models."""
        print(f"Accuracy: {self.accuracy}")
        print(f"Precision: {self.precision}")
        print(f"Recall: {self.recall}")
        print(f"F1 Score: {self.f1_score}")
        print(f"ROC AUC: {self.roc_auc}")
        print(f"AIC: {self.aic}")
        print(f"Deviance: {self.deviance}")
