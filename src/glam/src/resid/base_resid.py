import pandas as pd
from typing import Protocol


class BaseResid(Protocol):
    """Protocol for calculating residuals."""

    def residuals(self) -> pd.Series: ...
