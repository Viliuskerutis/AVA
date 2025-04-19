from sklearn.tree import DecisionTreeRegressor
import numpy as np
from config import REGRESSOR_DECISION_TREE_PKL_PATH
from price_prediction.regressors.base_regressor import BaseRegressor


class DecisionTreeCustomRegressor(BaseRegressor):
    """Decision Tree Regressor."""

    def __init__(self, max_depth: int = None, random_state: int = 42, **kwargs):
        super().__init__(max_depth=max_depth, random_state=random_state, **kwargs)
        self.model = DecisionTreeRegressor(
            max_depth=max_depth, random_state=random_state, **kwargs
        )
        self.path = REGRESSOR_DECISION_TREE_PKL_PATH

    def clear_fit(self):
        self.model = DecisionTreeRegressor(**self.params)

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        self.model.fit(X_train, y_train)
        print(f"Tree depth after fitting: {self.model.tree_.max_depth}")

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        return self.model.predict(X_test)
