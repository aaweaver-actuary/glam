from typing import Protocol, Generator

from glam.src.fitted_model.base_fitted_model import BaseFittedModel


__all__ = ["BaseModelList"]


class BaseModelList(Protocol):
    """Define a protocol interface for a list of models.

    This protocol defines the methods and attributes for a list of models. The list of models is functionally a container
    for models that can be iterated over, indexed, and added to.

    Attributes
    ----------
    models : list[BaseFittedModel]
        The list of models.
    model_generator : Generator[BaseFittedModel, None, None]
        A generator of models from the list of models.
    model_lookup : dict[int, BaseFittedModel]
        A dictionary of model indices to models.
    model : BaseFittedModel
        The model at the last index.

    Methods
    -------
    **add_model(model: BaseFittedModel) -> None**

        Add a fitted model to the list of models.

    **reset_models() -> None**

        Reset the list of models.

    **get_model(index: int) -> BaseFittedModel**

        Return the model at the given index.
    """

    @property
    def models(self) -> list[BaseFittedModel]: ...

    @models.setter
    def models(self, model: BaseFittedModel) -> None: ...

    @property
    def model_generator(self) -> Generator[BaseFittedModel, None, None]: ...

    @property
    def model_lookup(self) -> dict[int, BaseFittedModel]: ...

    @property
    def model(self) -> BaseFittedModel: ...

    def get_model(self, index: int) -> BaseFittedModel: ...

    def add_model(self, model: BaseFittedModel) -> None: ...

    def reset_models(self) -> None: ...
