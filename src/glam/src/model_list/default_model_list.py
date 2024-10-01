"""Implement the DefaultModelList class."""

from __future__ import annotations
from glam.src.fitted_model.base_fitted_model import BaseFittedModel
from typing import Generator


class DefaultModelList:
    """Implement the BaseModelList protocol to store a list of models.

    This class is a container for a list of models. It provides methods to add, reset, and retrieve models.

    Attributes
    ----------
    models : list[BaseFittedModel]
        The list of models.
    model_generator : Generator[BaseFittedModel, None, None]
        A generator of models from the list of models.
    model_lookup : dict[int, BaseFittedModel]
        A dictionary of model indices to models.
    model : BaseFittedModel
        The last model in the list.

    Methods
    -------
    **add_model(model: BaseFittedModel) -> None**

        Add a fitted model to the list of models.

    **reset_models() -> None**

            Reset the list of models.

    **get_model(index: int) -> BaseFittedModel**

        Return the model at the given index.

    """

    def __init__(self, models: list[BaseFittedModel] | None = None):
        self._models = models if models is not None else []

    def __repr__(self) -> str:  # noqa: D105
        return f"DefaultModelList(n_models={len(self.models)})"

    def __str__(self) -> str:  # noqa: D105
        return self.__repr__()

    @property
    def models(self) -> list[BaseFittedModel]:
        """Return the list of models.

        Returns
        -------
        list[BaseFittedModel]
            A list of BaseFittedModel objects.
        """
        return self._models

    @models.setter
    def models(self, model: BaseFittedModel) -> None:
        self._models.append(model)

    @property
    def model_generator(self) -> Generator[BaseFittedModel, None, None]:
        """Return a generator of models from the ordered list of models.

        Yields
        ------
        BaseFittedModel
            A model from the list of models.
        """
        for model in self.models:
            yield model

    @property
    def model_lookup(self) -> dict[int, BaseFittedModel]:
        """Return a dictionary of model indices to models.

        Returns
        -------
        dict[int, BaseFittedModel]
            A dictionary of model indices to models.
        """
        return dict(enumerate(self.models))

    @property
    def model(self) -> BaseFittedModel:
        """Return the last model in the list."""
        return self.models[-1]

    def add_model(self, model: BaseFittedModel) -> None:
        """Add a model to the list of models.

        Parameters
        ----------
        model : BaseFittedModel
            The model to add to the list of models.
        """
        current_models = self.models
        new_models = [*current_models, model]
        self._models = new_models

    def reset_models(self) -> None:
        """Reset the list of models.

        This method sets the list of models to an empty list.
        """
        self._models = []

    def get_model(self, index: int) -> BaseFittedModel:
        """Return the model at the given index.

        Parameters
        ----------
        index : int
            The index of the model to return.

        Returns
        -------
        BaseFittedModel
            The model at the given index.

        Raises
        ------
        IndexError
            If the index is out of range.
        """
        return self.model_lookup[index]
