import numpy as np
from sklearn.ensemble import RandomForestRegressor

from price_prediction.regressors.base_regressor import BaseRegressor


class RandomForestCustomRegressor(BaseRegressor):
    """Random Forest Regressor."""

    def __init__(self, n_estimators: int = 10, **kwargs):
        super().__init__(n_estimators=n_estimators, **kwargs)

        self.model = RandomForestRegressor(
            n_estimators=n_estimators, random_state=42, **kwargs
        )

    def clear_fit(self):
        self.model = RandomForestRegressor(**self.params)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        return self.model.predict(X_test)
