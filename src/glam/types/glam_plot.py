"""Define a union type for all backends used to make plots."""

from typing import Union

import plotly.graph_objects  # noqa: ICN001
import matplotlib.pyplot  # noqa: ICN001

GlamPlot = Union[plotly.graph_objects.Figure, matplotlib.pyplot.Figure]
