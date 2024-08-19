from typing import Generator, Protocol

from hit_ratio.old_glm.model_result import BaseModelResult


class BaseModelList(Protocol):
    @property
    def models(self) -> list[BaseModelResult]: ...

    @models.setter
    def models(self, model: BaseModelResult) -> None: ...

    @property
    def model_generator(self) -> Generator[BaseModelResult, None, None]: ...

    @property
    def model_lookup(self) -> dict[int, BaseModelResult]: ...

    @property
    def model(self) -> BaseModelResult: ...

    def get_model(self, index: int) -> BaseModelResult: ...

    def add_model(self, model: BaseModelResult) -> None: ...

    def reset_models(self) -> None: ...


class DefaultModelList:
    def __init__(self, models: list[BaseModelResult] | None = None):
        self._models = models if models is not None else []

    @property
    def models(self) -> list[BaseModelResult]:
        return self._models

    @models.setter
    def models(self, model: BaseModelResult) -> None:
        self._models.append(model)

    def add_model(self, model: BaseModelResult) -> None:
        """Add a model to the list of models.

        Parameters
        ----------
        model : BaseModelResult
            The model to add to the list of models.
        """
        current_models = self.models
        new_models = current_models + [model]
        self._models = new_models

    def reset_models(self) -> None:
        """Reset the list of models.

        This method sets the list of models to an empty list.
        """
        self._models = []

    @property
    def model_generator(self) -> Generator[BaseModelResult, None, None]:
        """Return a generator of models from the ordered list of models.

        Yields
        ------
        BaseModelResult
            A model from the list of models.
        """
        for model in self.models:
            yield model

    @property
    def model_lookup(self) -> dict[int, BaseModelResult]:
        """Return a dictionary of model indices to models.

        Returns
        -------
        dict[int, BaseModelResult]
            A dictionary of model indices to models.
        """
        return {index: model for index, model in enumerate(self.models)}

    def get_model(self, index: int) -> BaseModelResult:
        """Return the model at the given index.

        Parameters
        ----------
        index : int
            The index of the model to return.

        Returns
        -------
        BaseModelResult
            The model at the given index.

        Raises
        ------
        IndexError
            If the index is out of range.
        """
        return self.model_lookup[index]

    @property
    def model(self) -> BaseModelResult:
        """Return the last model in the list."""
        return self.models[-1]
