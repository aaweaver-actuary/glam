import numpy as np

from abc import ABC, abstractmethod


class BaseHatMatrixCalculator(ABC):
    @abstractmethod
    def X(self) -> np.ndarray:
        pass

    @abstractmethod
    def weight_matrix(self) -> np.ndarray:
        pass

    def calculate(self) -> np.ndarray:
        """Calculate the hat matrix."""
        # Calculation is split into three terms for readability:
        # H = X(X'WX)^{-1}X'W
        # term1 = X
        # term2 = (X'WX)^{-1}
        # term3 = X'W

        term1 = self.X()
        term2 = np.linalg.inv(self.X().T @ self.weight_matrix() @ self.X())
        term3 = self.X().T @ self.weight_matrix()

        return term1 @ term2 @ term3
