from typing import Protocol
from glam.src.data.base_model_data import BaseModelData
from glam.src.fitters.base_model_fitter import BaseModelFitter
from glam.src.data.data_prep import BaseDataSplitter, BasePreprocessor
from glam.src.model_list.base_model_list import BaseModelList

__all__ = ["BaseAnalysis"]


class BaseAnalysis(Protocol):
    def __init__(
        self,
        data: BaseModelData,
        features: list[str],
        fitter: BaseModelFitter,
        splitter: BaseDataSplitter,
        preprocessor: BasePreprocessor,
        models: BaseModelList,
    ) -> None: ...
