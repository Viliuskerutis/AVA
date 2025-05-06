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
from price_prediction.regressors.densenet_regressor import DenseNetRegressor
from price_prediction.regressors.neural_midpoint_network_regressor import (
    NeuralMidpointNetworkRegressor,
)
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
        df = self.preprocess_data(df, is_training=True)
        X = self.generate_combined_embeddings(df)
        y = df["Sold Price"].astype(float)
        midpoints = self.calculate_midpoints(df)

        for regressor in self.regressors:
            regressor.feature_columns = df.columns.tolist()
            regressor.feature_types = df.dtypes.to_dict()

            if isinstance(regressor, NeuralMidpointNetworkRegressor):
                (
                    X_train,
                    X_validation,
                    y_train,
                    y_validation,
                    midpoints_train,
                    midpoints_validation,
                ) = train_test_split(X, y, midpoints, test_size=0.2, random_state=42)
                regressor.train(
                    X_train,
                    y_train,
                    X_validation,
                    y_validation,
                    midpoints_train,
                    midpoints_validation,
                )
            elif isinstance(regressor, NeuralNetworkRegressor) or isinstance(
                regressor, DenseNetRegressor
            ):
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
        self.meta_regressor.fit(meta_features, y)
        print("Meta regressor fitted:", hasattr(self.meta_regressor, "coef_"))

    def train_with_same_data(
        self, df: pd.DataFrame, log_results: bool = True
    ) -> pd.DataFrame:
        for regressor in self.regressors:
            regressor.clear_fit()

        processed_df = self.preprocess_data(df, is_training=True)

        for regressor in self.regressors:
            regressor.feature_columns = processed_df.columns.tolist()
            regressor.feature_types = processed_df.dtypes.to_dict()

        X, y = self.generate_combined_embeddings(processed_df), processed_df[
            "Sold Price"
        ].astype(float)

        for regressor in self.regressors:
            regressor.train(X, y)

        base_predictions = np.column_stack(
            [regressor.predict(X) for regressor in self.regressors]
        )

        if not hasattr(self.meta_regressor, "coef_"):
            self.meta_regressor.fit(base_predictions, y)
            print("Meta regressor has been fitted.")

        y_pred = self.meta_regressor.predict(base_predictions)

        # Include Estimated Min and Max Prices in the results DataFrame
        result_df = pd.DataFrame(
            {
                "Actual": y,
                "Predicted": y_pred,
                "Estimated Minimum Price": processed_df["Estimated Minimum Price"],
                "Estimated Maximum Price": processed_df["Estimated Maximum Price"],
            }
        )

        metrics = self.evaluate_metrics(y, y_pred, processed_df)

        if log_results:
            text_columns = self.TEXT_COLUMNS + self.additional_text_columns
            numeric_columns = self.NUMERIC_COLUMNS + self.additional_numeric_columns

            used_columns_count = len(text_columns) + len(numeric_columns)

            text_columns_str = ";".join(text_columns)
            numeric_columns_str = ";".join(numeric_columns)

            df_size = (len(processed_df), used_columns_count)

            regressor_names = [
                regressor.__class__.__name__ for regressor in self.regressors
            ]

            predictor_info = {
                "text_columns": text_columns_str,
                "numeric_columns": numeric_columns_str,
                "max_missing_percent": self.max_missing_percent,
                "use_separate_numeric_features": self.use_separate_numeric_features,
                "encode_per_column": self.encode_per_column,
                "hot_encode_columns": (
                    ";".join(self.hot_encode_columns) if self.hot_encode_columns else ""
                ),
                "use_artfacts": self.use_artfacts,
                "use_images": self.use_images,
                "use_count": self.use_count,
                "use_synthetic": self.use_synthetic,
                "use_artsy": self.use_artsy,
                "embedding_model_type": (
                    self.embedding_model_type.name
                    if hasattr(self.embedding_model_type, "name")
                    else str(self.embedding_model_type)
                ),
                "test_size": 0.0,  # No test split
                "meta_regressor": self.meta_regressor.__class__.__name__,
            }

            self.log_results(
                metrics,
                predictor_info,
                regressor_names,
                df_size,
                None,
            )

        return result_df

    def train_with_test_split(
        self, df: pd.DataFrame, test_size: float, log_results: bool = True
    ) -> pd.DataFrame:
        for regressor in self.regressors:
            regressor.clear_fit()

        processed_df = self.preprocess_data(df, is_training=True)
        X, y = self.generate_combined_embeddings(processed_df), processed_df[
            "Sold Price"
        ].astype(float)
        midpoints = self.calculate_midpoints(processed_df)
        X_train, X_test, y_train, y_test, midpoints_train, midpoints_test = (
            train_test_split(X, y, midpoints, test_size=test_size, random_state=42)
        )

        nn_info = None
        for regressor in self.regressors:
            regressor.feature_columns = processed_df.columns.tolist()
            regressor.feature_types = processed_df.dtypes.to_dict()

            if isinstance(regressor, NeuralMidpointNetworkRegressor):
                if nn_info is None:
                    nn_info = {
                        "model_class": regressor.model_class.__name__,
                        "loss_function": regressor.loss_function.__class__.__name__,
                        "hidden_units": regressor.hidden_units,
                    }
                else:
                    nn_info["model_class"] += ";" + regressor.model_class.__name__
                    nn_info["loss_function"] += ";" + (
                        regressor.loss_function.__class__.__name__
                    )
                    nn_info["hidden_units"] = (
                        nn_info["hidden_units"] + ";" + regressor.hidden_units
                    )

                (
                    X_validation,
                    X_test,
                    y_validation,
                    y_test,
                    midpoints_validation,
                    midpoints_test,
                ) = train_test_split(
                    X_test, y_test, midpoints_test, test_size=0.6, random_state=42
                )

                regressor.train(
                    X_train,
                    y_train,
                    X_validation,
                    y_validation,
                    midpoints_train,
                    midpoints_validation,
                )
            elif isinstance(regressor, NeuralNetworkRegressor) or isinstance(
                regressor, DenseNetRegressor
            ):
                if nn_info is None:
                    nn_info = {
                        "model_class": regressor.model_class.__name__,
                        "loss_function": regressor.loss_function.__class__.__name__,
                        "hidden_units": regressor.hidden_units,
                    }
                else:
                    nn_info["model_class"] += ";" + regressor.model_class.__name__
                    nn_info["loss_function"] += ";" + (
                        regressor.loss_function.__class__.__name__
                    )
                    nn_info["hidden_units"] = (
                        nn_info["hidden_units"] + ";" + regressor.hidden_units
                    )

                X_validation, X_test, y_validation, y_test = train_test_split(
                    X_test, y_test, test_size=0.6, random_state=42
                )
                regressor.train(X_train, y_train, X_validation, y_validation)
            else:
                regressor.train(X_train, y_train)

        y_pred = np.column_stack(
            [regressor.predict(X_test) for regressor in self.regressors]
        )
        if not hasattr(self.meta_regressor, "coef_"):
            self.meta_regressor.fit(y_pred, y_test)
            print("Meta regressor has been fitted.")

        y_pred = self.meta_regressor.predict(y_pred)

        df_test = processed_df.iloc[y_test.index]

        # Include Estimated Min and Max Prices in the results DataFrame
        result_df = pd.DataFrame(
            {
                "Artist Name": df_test["Artist name"].values,
                "Painting Name": df_test["Painting name"].values,
                "Actual": y_test.values,
                "Predicted": y_pred,
                "Estimated Minimum Price": df_test["Estimated Minimum Price"].values,
                "Estimated Maximum Price": df_test["Estimated Maximum Price"].values,
            }
        )

        metrics = self.evaluate_metrics(y_test, y_pred, df_test)

        if log_results:
            text_columns = self.TEXT_COLUMNS + self.additional_text_columns
            numeric_columns = self.NUMERIC_COLUMNS + self.additional_numeric_columns

            used_columns_count = len(text_columns) + len(numeric_columns)

            text_columns_str = ";".join(text_columns)
            numeric_columns_str = ";".join(numeric_columns)

            df_size = (len(processed_df), used_columns_count)

            regressor_names = [
                regressor.__class__.__name__ for regressor in self.regressors
            ]

            predictor_info = {
                "text_columns": text_columns_str,
                "numeric_columns": numeric_columns_str,
                "max_missing_percent": self.max_missing_percent,
                "use_separate_numeric_features": self.use_separate_numeric_features,
                "encode_per_column": self.encode_per_column,
                "hot_encode_columns": (
                    ";".join(self.hot_encode_columns) if self.hot_encode_columns else ""
                ),
                "use_artfacts": self.use_artfacts,
                "use_images": self.use_images,
                "use_count": self.use_count,
                "use_synthetic": self.use_synthetic,
                "use_artsy": self.use_artsy,
                "embedding_model_type": (
                    self.embedding_model_type.name
                    if hasattr(self.embedding_model_type, "name")
                    else str(self.embedding_model_type)
                ),
                "test_size": test_size,
                "meta_regressor": self.meta_regressor.__class__.__name__,
            }

            self.log_results(
                metrics,
                predictor_info,
                regressor_names,
                df_size,
                nn_info if nn_info else None,
            )

        return result_df

    def predict_single_painting(
        self, row: Union[Dict[str, Any], pd.Series]
    ) -> Optional[float]:
        if isinstance(row, dict):
            df_single = pd.DataFrame([row])
        elif isinstance(row, pd.Series):
            df_single = row.to_frame().T
        else:
            raise ValueError("Input row must be a dictionary or a pandas Series")

        processed_df = self.preprocess_data(df_single, is_training=False)
        processed_df = self.filter_prediction_columns(processed_df)
        X = self.generate_combined_embeddings(processed_df)
        base_predictions = np.column_stack(
            [regressor.predict(X) for regressor in self.regressors]
        )
        return self.meta_regressor.predict(base_predictions)[0]

    def predict_random_paintings(self, df: pd.DataFrame, count: int):
        processed_df = self.preprocess_data(df, is_training=True)

        X, y = self.generate_combined_embeddings(processed_df), processed_df[
            "Sold Price"
        ].astype(float)

        np.random.seed(42)
        random_indexes = np.random.randint(0, len(X), size=count)
        for index in random_indexes:
            X_random = X[index].reshape(1, -1)
            y_random = y.iloc[index]

            y_pred = np.column_stack(
                [regressor.predict(X_random) for regressor in self.regressors]
            )
            y_pred = self.meta_regressor.predict(y_pred)[0]
            print(f"Actual: {y_random}, Predicted: {y_pred}")

    def cross_validate(self, df: pd.DataFrame, k: int = 5) -> Dict[str, float]:
        df = self.preprocess_data(df, is_training=True)
        X, y = self.generate_combined_embeddings(df), df["Sold Price"].astype(float)

        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        metrics = []

        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            for regressor in self.regressors:
                regressor.clear_fit()
                if isinstance(regressor, NeuralNetworkRegressor):
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

            base_test_preds = np.column_stack(
                [regressor.predict(X_test) for regressor in self.regressors]
            )

            if not hasattr(self.meta_regressor, "coef_"):
                self.meta_regressor.fit(base_test_preds, y_test)

            y_pred = self.meta_regressor.predict(base_test_preds)

            fold_metrics = {
                "MAE": mean_absolute_error(y_test, y_pred),
                "MSE": mean_squared_error(y_test, y_pred),
                "MAPE": mean_absolute_percentage_error(y_test, y_pred),
                "R2": r2_score(y_test, y_pred),
            }
            metrics.append(fold_metrics)

        avg_metrics = {key: np.mean([m[key] for m in metrics]) for key in metrics[0]}
        print("Cross-Validation Metrics:", avg_metrics)
        return avg_metrics

    def filter_prediction_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self.regressors[0], "feature_columns") or not hasattr(
            self.regressors[0], "feature_types"
        ):
            raise ValueError(
                "Feature columns or types are not stored in the regressor. "
                "Ensure the model is trained or loaded correctly."
            )

        unexpected_columns = set(df.columns) - set(self.regressors[0].feature_columns)
        if unexpected_columns:
            print(f"Warning: Unexpected columns in input data: {unexpected_columns}")

        new_columns = {}
        for col in self.regressors[0].feature_columns:
            if col not in df.columns:
                expected_dtype = self.regressors[0].feature_types[col]
                if pd.api.types.is_numeric_dtype(expected_dtype):
                    if col.startswith("Surface_") or col.startswith("Materials_"):
                        new_columns[col] = 0
                    else:
                        new_columns[col] = -1
                    if col not in self.NUMERIC_COLUMNS:
                        self.additional_numeric_columns.append(col)
                elif pd.api.types.is_bool_dtype(expected_dtype):
                    new_columns[col] = False
                else:
                    new_columns[col] = "Unknown"
                    if col not in self.TEXT_COLUMNS:
                        self.additional_text_columns.append(col)

        if new_columns:
            df = pd.concat([df, pd.DataFrame(new_columns, index=df.index)], axis=1)

        df = df[self.regressors[0].feature_columns]

        for col in self.regressors[0].feature_columns:
            df[col] = df[col].astype(self.regressors[0].feature_types[col])

        return df
