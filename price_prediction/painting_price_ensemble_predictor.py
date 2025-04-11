from typing import Dict, Any, List, Optional, Union
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split, KFold
from helpers.file_manager import FileManager
from price_prediction.base_price_predictor import BasePredictor
from price_prediction.embedding_model_type import EmbeddingModelType
from price_prediction.regressors.base_regressor import BaseRegressor
from price_prediction.regressors.neural_network_regressor import NeuralNetworkRegressor


class PaintingPriceEnsemblePredictor(BasePredictor):
    """
    Handles painting price prediction with preprocessing, feature extraction, and regressor selection.
    """

    def __init__(
        self,
        regressors: List[BaseRegressor],
        meta_regressor: BaseRegressor,
        max_missing_percent: float,
        use_separate_numeric_features: bool,
        encode_per_column: bool,
        hot_encode_columns: List[str],
        use_artfacts: bool,
        use_images: int,
        use_count: bool,
        use_synthetic: bool,
        use_artsy: bool,
        embedding_model_type: EmbeddingModelType,
        artist_info_path: str,
        image_features_path: str,
        artist_count_path: str,
        synthetic_paths: List[str],
        artsy_path: str,
    ):
        super().__init__(
            max_missing_percent=max_missing_percent,
            use_separate_numeric_features=use_separate_numeric_features,
            encode_per_column=encode_per_column,
            hot_encode_columns=hot_encode_columns,
            use_artfacts=use_artfacts,
            use_images=use_images,
            use_count=use_count,
            use_synthetic=use_synthetic,
            use_artsy=use_artsy,
            embedding_model_type=embedding_model_type,
            artist_info_path=artist_info_path,
            image_features_path=image_features_path,
            artist_count_path=artist_count_path,
            synthetic_paths=synthetic_paths,
            artsy_path=artsy_path,
        )
        self.regressors = regressors
        self.meta_regressor = meta_regressor

    def save_model(self, model_path: str) -> None:
        data_to_save = {
            "regressors": self.regressors,
            "meta_regressor": self.meta_regressor,
            # Assuming all regressors have the same feature columns
            "feature_columns": self.regressors[0].feature_columns,
            "feature_types": self.regressors[0].feature_types,
        }
        FileManager.save_to_pkl(data_to_save, model_path)

    def load_model(self, model_path: str) -> bool:
        loaded_data = FileManager.load_from_pkl(model_path)
        if loaded_data:
            self.regressors = loaded_data["regressors"]
            self.meta_regressor = loaded_data["meta_regressor"]
            self.regressors[0].feature_columns = loaded_data["feature_columns"]
            self.regressors[0].feature_types = loaded_data["feature_types"]
            return True
        return False

    def train(self, df: pd.DataFrame) -> None:
        for regressor in self.regressors:
            regressor.clear_fit()
        df = self.preprocess_data(df)
        X = self.generate_combined_embeddings(df)
        y = df["Sold Price"].astype(float)

        for regressor in self.regressors:
            regressor.feature_columns = df.columns.tolist()
            regressor.feature_types = df.dtypes.to_dict()

            if isinstance(regressor, NeuralNetworkRegressor):
                X_train, X_validation, y_train, y_validation = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
                regressor.train(X_train, y_train, X_validation, y_validation)
            else:
                regressor.train(X, y)

        self.meta_regressor = LinearRegression()
        meta_features = np.column_stack(
            [regressor.predict(X) for regressor in self.regressors]
        )
        print("Meta features shape:", meta_features.shape)
        self.meta_regressor.fit(meta_features, y)
        print("Meta regressor fitted:", hasattr(self.meta_regressor, "coef_"))

    def predict_single_painting(
        self, row: Union[Dict[str, Any], pd.Series]
    ) -> Optional[float]:
        # Convert input into a single-row DataFrame.
        if isinstance(row, dict):
            df_single = pd.DataFrame([row])
        elif isinstance(row, pd.Series):
            df_single = row.to_frame().T
        else:
            raise ValueError("Input row must be a dictionary or a pandas Series")

        processed_df = self.preprocess_data(df_single)
        processed_df = self.filter_prediction_columns(processed_df)
        X = self.generate_combined_embeddings(processed_df)
        base_predictions = np.column_stack(
            [regressor.predict(X) for regressor in self.regressors]
        )
        return self.meta_regressor.predict(base_predictions)[0]

    def train_with_test_split(self, df: pd.DataFrame, test_size: float) -> pd.DataFrame:
        for regressor in self.regressors:
            regressor.clear_fit()

        # Preprocess the data
        df = self.preprocess_data(df)

        X, y = self.generate_combined_embeddings(df), df["Sold Price"].astype(float)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        # Train the base regressors
        for regressor in self.regressors:
            regressor.feature_columns = df.columns.tolist()
            regressor.feature_types = df.dtypes.to_dict()

            if isinstance(regressor, NeuralNetworkRegressor):
                X_validation, X_test, y_validation, y_test = train_test_split(
                    X_test, y_test, test_size=0.6, random_state=42
                )
                regressor.train(X_train, y_train, X_validation, y_validation)
            else:
                regressor.train(X_train, y_train)

        base_test_preds = np.column_stack(
            [regressor.predict(X_test) for regressor in self.regressors]
        )

        if not hasattr(self.meta_regressor, "coef_"):
            self.meta_regressor.fit(base_test_preds, y_test)
            print("Meta regressor has been fitted.")

        y_pred = self.meta_regressor.predict(base_test_preds)

        # Evaluate the model performance
        print(f"MAE: {mean_absolute_error(y_test, y_pred)}")
        print(f"MSE: {mean_squared_error(y_test, y_pred)}")
        print(f"MAPE: {mean_absolute_percentage_error(y_test, y_pred)}")
        print(f"R2: {r2_score(y_test, y_pred)}")

        # Return the results as a DataFrame
        return pd.DataFrame({"Actual": y_test, "Predicted": y_pred})

    def cross_validate(self, df: pd.DataFrame, k: int = 5) -> Dict[str, float]:
        # Preprocess the data
        df = self.preprocess_data(df)
        X, y = self.generate_combined_embeddings(df), df["Sold Price"].astype(float)

        # Initialize k-fold cross-validation
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        metrics = []

        for train_index, test_index in kf.split(X):
            # Split data into training and testing sets
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            # Clear and train base regressors
            for regressor in self.regressors:
                regressor.clear_fit()
                if isinstance(regressor, NeuralNetworkRegressor):
                    # Split validation set for neural networks
                    X_train_split, X_validation, y_train_split, y_validation = (
                        train_test_split(
                            X_train, y_train, test_size=0.2, random_state=42
                        )
                    )
                    regressor.train(
                        X_train_split, y_train_split, X_validation, y_validation
                    )
                else:
                    regressor.train(X_train, y_train)

            # Generate predictions from base regressors
            base_test_preds = np.column_stack(
                [regressor.predict(X_test) for regressor in self.regressors]
            )

            # Train meta-regressor if not already fitted
            if not hasattr(self.meta_regressor, "coef_"):
                self.meta_regressor.fit(base_test_preds, y_test)

            # Predict using the meta-regressor
            y_pred = self.meta_regressor.predict(base_test_preds)

            # Compute metrics for this fold
            fold_metrics = {
                "MAE": mean_absolute_error(y_test, y_pred),
                "MSE": mean_squared_error(y_test, y_pred),
                "MAPE": mean_absolute_percentage_error(y_test, y_pred),
                "R2": r2_score(y_test, y_pred),
            }
            metrics.append(fold_metrics)

        # Compute average metrics across all folds
        avg_metrics = {key: np.mean([m[key] for m in metrics]) for key in metrics[0]}
        print("Cross-Validation Metrics:", avg_metrics)
        return avg_metrics

    def filter_prediction_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        # Verify that the regressor holds the necessary schema.
        if not hasattr(self.regressors[0], "feature_columns") or not hasattr(
            self.regressors[0], "feature_types"
        ):
            raise ValueError(
                "Feature columns or types are not stored in the regressor. "
                "Ensure the model is trained or loaded correctly."
            )

        # Add logging if different training/prediction weights used
        unexpected_columns = set(df.columns) - set(self.regressors[0].feature_columns)
        if unexpected_columns:
            print(f"Warning: Unexpected columns in input data: {unexpected_columns}")

        # Add missing columns with default values and update additional columns
        for col in self.regressors[0].feature_columns:
            if col not in df.columns:
                expected_dtype = self.regressors[0].feature_types[col]
                if pd.api.types.is_numeric_dtype(expected_dtype):
                    df[col] = -1
                    if col not in self.NUMERIC_COLUMNS:
                        self.additional_numeric_columns.append(col)
                else:
                    df[col] = "Unknown"
                    if col not in self.TEXT_COLUMNS:
                        self.additional_text_columns.append(col)

        # Keep only the expected columns and reorder them.
        df = df[self.regressors[0].feature_columns]

        # Cast each column to its expected type.
        for col in self.regressors[0].feature_columns:
            df[col] = df[col].astype(self.regressors[0].feature_types[col])

        return df
