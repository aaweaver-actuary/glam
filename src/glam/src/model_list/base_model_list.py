from typing import Protocol, Generator

from glam.src.fitted_model.base_fitted_model import BaseFittedModel


class BaseModelList(Protocol):
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


