import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from config import REGRESSOR_HISTOGRAM_GRADIENT_BOOSTING_PKL_PATH
from price_prediction.regressors.base_regressor import BaseRegressor


class HistogramGradientBoostingRegressor(BaseRegressor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.model = HistGradientBoostingRegressor(**kwargs)
        self.path = REGRESSOR_HISTOGRAM_GRADIENT_BOOSTING_PKL_PATH

    def clear_fit(self):
        self.model = HistGradientBoostingRegressor(**self.params)

    def train(self, X_train, y_train):
        print("Training...")
        self.model.fit(X_train, y_train)

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        return self.model.predict(X_test)
