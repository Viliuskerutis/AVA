import numpy as np
from lightgbm import LGBMRegressor
import pandas as pd

from config import REGRESSOR_LIGHTGBM_PKL_PATH
from price_prediction.regressors.base_regressor import BaseRegressor


class LightGBMRegressor(BaseRegressor):
    """LightGBM Regressor."""

    def __init__(self, n_estimators=500, **kwargs):
        super().__init__(n_estimators=n_estimators, **kwargs)

        self.model = LGBMRegressor(n_estimators=n_estimators, random_state=42, **kwargs)
        self.path = REGRESSOR_LIGHTGBM_PKL_PATH

    def clear_fit(self):
        self.model = LGBMRegressor(**self.params)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train, eval_metric="mae")

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        X_test = pd.DataFrame(
            X_test, columns=self.model.feature_name_
        )  # LightGBM requires names
        return self.model.predict(X_test)
