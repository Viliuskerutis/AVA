import numpy as np
from scipy.optimize import minimize


class MAPEMetaRegressor:
    """
    Custom meta-regressor that minimizes Mean Absolute Percentage Error (MAPE).
    """

    def __init__(self):
        self.weights = None

    def _mape_loss(self, weights, base_predictions, y_true):
        """
        Compute MAPE loss for given weights.
        """
        # Weighted predictions
        weighted_preds = np.dot(base_predictions, weights)
        # Avoid division by zero
        y_true = np.where(y_true == 0, 1e-8, y_true)
        # Compute MAPE
        mape = np.mean(np.abs((y_true - weighted_preds) / y_true))
        return mape

    def fit(self, base_predictions, y_true):
        """
        Fit the meta-regressor by optimizing weights to minimize MAPE.
        """
        n_regressors = base_predictions.shape[1]
        # Initialize weights equally
        initial_weights = np.ones(n_regressors) / n_regressors

        # Constraints: weights must sum to 1
        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
        # Bounds: weights must be between 0 and 1
        bounds = [(0, 1) for _ in range(n_regressors)]

        # Optimize weights to minimize MAPE
        result = minimize(
            self._mape_loss,
            initial_weights,
            args=(base_predictions, y_true),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        if result.success:
            self.weights = result.x
        else:
            raise ValueError("Optimization failed: " + result.message)

    def predict(self, base_predictions):
        """
        Predict using the optimized weights.
        """
        if self.weights is None:
            raise ValueError("The meta-regressor has not been fitted yet.")
        return np.dot(base_predictions, self.weights)
