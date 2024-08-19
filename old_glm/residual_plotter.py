from typing import Protocol

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from hit_ratio.old_glm.residual_calculator import BaseResidualCalculator


class BaseResidualPlotter(Protocol):
    """Define a protocol interface for plotting residuals."""

    def plot(self, calculator: BaseResidualCalculator) -> go.Figure: ...
