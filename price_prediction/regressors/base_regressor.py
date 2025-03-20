from abc import ABC, abstractmethod
from typing import Any
import numpy as np

from helpers.file_manager import FileManager


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

    def save_model(self, file_path: str) -> None:
        """Saves the trained model to a .pkl file."""
        FileManager.save_to_pkl(self.model, file_path)

    def load_model(self, file_path: str) -> bool:
        """
        Loads the model from a .pkl file.

        Returns:
            bool: True if the model was successfully loaded, False otherwise.
        """
        loaded_model = FileManager.load_from_pkl(file_path)
        if loaded_model is not None:
            self.model = loaded_model
            return True
        return False
