"""Define a base class for calculating the hat matrix."""

import numpy as np


class BaseHatMatrixCalculator:
    """Abstract base class for calculating the hat matrix."""

    @property
    def X(self) -> np.ndarray:
        """Return the feature matrix."""
        raise NotImplementedError

    @property
    def W(self) -> np.ndarray:
        """Return the weight matrix."""
        raise NotImplementedError

    def calculate(self) -> np.ndarray:
        """Calculate the hat matrix.

        The hat matrix is defined as H = X(X'WX)^{-1}X'W, where:
        - X is the design matrix
        - W is the weight matrix
        - H is the hat matrix

        My implementation splits the calculation into three terms for readability:
        - term1 = X
        - term2 = (X'WX)^{-1}
        - term3 = X'W

        Thus, the hat matrix is calculated as term1 @ term2 @ term3.
        """
        term1 = self.X
        term2 = np.linalg.inv(self.X.T @ self.W @ self.X)
        term3 = self.X.T @ self.W

        return term1 @ term2 @ term3
