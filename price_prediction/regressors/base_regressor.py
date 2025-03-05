from abc import ABC, abstractmethod
from typing import Any
import numpy as np


class BaseRegressor(ABC):
    """Abstract base class for all regressors."""

    def __init__(self, **regressor_params: Any):
        self.params = regressor_params

    @abstractmethod
    def clear_fit() -> None:
        pass

    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Train method that must be implemented by all regressors."""
        pass

    @abstractmethod
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Predict method that must be implemented by all regressors."""
        pass
