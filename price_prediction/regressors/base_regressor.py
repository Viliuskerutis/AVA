from abc import ABC, abstractmethod
from typing import Any
import numpy as np

from helpers.file_manager import FileManager


class BaseRegressor(ABC):
    """Abstract base class for all regressors."""

    def __init__(self, **regressor_params: Any):
        self.params = regressor_params
        self.feature_columns = []
        self.feature_types = []

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
        """
        Saves the trained model along with feature column names and types to a .pkl file.
        """
        data_to_save = {
            "model": self.model,
            "feature_columns": self.feature_columns,
            "feature_types": self.feature_types,
        }
        FileManager.save_to_pkl(data_to_save, file_path)

    def load_model(self, file_path: str) -> bool:
        """
        Loads the model, feature columns, and feature types from a .pkl file.

        Returns:
            bool: True if loaded successfully, False otherwise.
        """
        loaded_data = FileManager.load_from_pkl(file_path)
        if loaded_data is not None:
            self.model = loaded_data.get("model")
            self.feature_columns = loaded_data.get("feature_columns")
            self.feature_types = loaded_data.get("feature_types")
            return True
        return False
