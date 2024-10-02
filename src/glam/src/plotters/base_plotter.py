"""Define an abstract base class for all plotter classes."""

from abc import ABC, abstractmethod
from glam.types import GlamPlot


class BasePlotter(ABC):
    """Define an abstract base class for all plotter classes."""

    @abstractmethod
    def cat_plot(self) -> GlamPlot:
        """Create a categorical plot."""
