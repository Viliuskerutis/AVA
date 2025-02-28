from price_prediction.base_regressor import BaseRegressor
import numpy as np
from sklearn.neighbors import KNeighborsRegressor


class KNNRegressor(BaseRegressor):
    """K-Nearest Neighbors Regressor."""

    def __init__(self, n_neighbors: int = 5, **kwargs):
        super().__init__(n_neighbors=n_neighbors, **kwargs)

        self.model = KNeighborsRegressor(n_neighbors=n_neighbors, **kwargs)

    def clear_fit(self):
        self.model = KNeighborsRegressor(**self.params)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        return self.model.predict(X_test)
