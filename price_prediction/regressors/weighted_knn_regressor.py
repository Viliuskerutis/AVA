import numpy as np
from sklearn.neighbors import KNeighborsRegressor

from config import REGRESSOR_KNN_PKL_PATH
from price_prediction.regressors.base_regressor import BaseRegressor


class WeightedKNNRegressor(BaseRegressor):
    """Weighted K-Nearest Neighbors Regressor that uses distance weighting."""

    def __init__(self, n_neighbors: int = 5, weights: str = "distance", **kwargs):
        super().__init__(n_neighbors=n_neighbors, weights=weights, **kwargs)

        self.model = KNeighborsRegressor(
            n_neighbors=n_neighbors,
            weights=weights,  # Use 'distance' for weighted predictions
            **kwargs
        )
        self.path = REGRESSOR_KNN_PKL_PATH

    def clear_fit(self):
        self.model = KNeighborsRegressor(**self.params)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        return self.model.predict(X_test)
