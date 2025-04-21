from abc import ABC, abstractmethod
from typing import Any, Dict, List

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from data_processing.data_filter_pipeline import (
    ensure_data_filled_and_correct,
    process_keep_relevant,
)
from helpers.file_manager import FileManager
from helpers.property_modifier import PropertyModifier
from price_prediction.embedding_model_type import EmbeddingModelType
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns


class BasePredictor(ABC):
    """
    Abstract base class for predictors to enforce common functionality.
    """

    NUMERIC_COLUMNS = [
        # "Artist Birth Year",
        # "Artist Death Year",
        # "Creation Year",
        "Width",
        "Height",
        "Years from auction till now",
        "Years from creation till auction",
        # "Years from birth till creation",
        "Artist Lifetime",
        "Average Sold Price",
        "Min Sold Price",
        "Max Sold Price",
    ]

    TEXT_COLUMNS = [
        "Artist name",
        # "Painting name",
        # "Auction name",
        "Auction House",
        "Auction Country",
        "Materials",
        "Surface",
        # "Is dead",
        "Signed",
    ]

    def __init__(
        self,
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
        artist_count_path: str,
        synthetic_paths: List[str],
        artsy_path: str,
        image_features_path: str,
    ):
        self.max_missing_percent = max_missing_percent
        self.use_separate_numeric_features = use_separate_numeric_features
        self.encode_per_column = encode_per_column
        self.hot_encode_columns = hot_encode_columns
        self.use_artfacts = use_artfacts
        self.use_images = use_images
        self.use_count = use_count
        self.use_synthetic = use_synthetic
        self.use_artsy = use_artsy
        self.embedding_model_type = embedding_model_type
        self.artist_info_path = artist_info_path
        self.artist_count_path = artist_count_path
        self.synthetic_paths = synthetic_paths
        self.artsy_path = artsy_path
        self.image_features_path = image_features_path

        self.model = SentenceTransformer(embedding_model_type.value)

        self.additional_text_columns = []
        self.additional_numeric_columns = []

    @abstractmethod
    def save_model(self, model_path: str) -> None:
        """
        Save the predictor's state (e.g., model weights, feature columns).
        """
        pass

    @abstractmethod
    def load_model(self, model_path: str) -> bool:
        """
        Load the predictor's state from a file.

        Returns:
            bool: True if the model was loaded successfully, False otherwise.
        """
        pass

    @abstractmethod
    def train(self, df) -> None:
        """
        Train the predictor using the provided dataset.

        Args:
            df (pd.DataFrame): The dataset to train the predictor on.
        """
        pass

    @abstractmethod
    def train_with_same_data(self, df) -> None:
        """
        Train the predictor using the same dataset.

        Args:
            df (pd.DataFrame): The dataset to train the predictor on.
        """
        pass

    @abstractmethod
    def train_with_test_split(self, df, test_size: float) -> None:
        """
        Train the predictor using a train-test split.

        Args:
            df (pd.DataFrame): The dataset to train the predictor on.
            test_size (float): The proportion of the dataset to include in the test split.
        """
        pass

    @abstractmethod
    def predict_single_painting(self, painting_data_dict: dict) -> float:
        """
        Predict the price of a single painting based on its data.

        Args:
            painting_data_dict (dict): Dictionary containing painting data.

        Returns:
            float: The predicted price of the painting.
        """
        pass

    @abstractmethod
    def predict_random_paintings(self, df: pd.DataFrame, count: int):
        """
        Predict the price of a random subset of paintings from the dataset.

        Args:
            df (pd.DataFrame): The dataset to use for predictions.
            count (int): Number of random paintings to predict.

        Returns:
            pd.DataFrame: DataFrame containing the predicted prices.
        """
        pass

    @abstractmethod
    def cross_validate(self, df: pd.DataFrame, k: int) -> None:
        """
        Perform k-fold cross-validation on the dataset.

        Args:
            df (pd.DataFrame): The dataset to use for cross-validation.
            k (int): Number of folds for cross-validation.
        """
        pass

    @abstractmethod
    def filter_prediction_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensures that the DataFrame df contains only the feature columns that were used during training.
        If a column is missing, it is added with a default value based on its type.
        Then, casts each column to its expected dtype.
        Uses schema stored on self.regressor.
        """
        pass

    def convert_columns_to_numeric(
        self, df: pd.DataFrame, columns: List[str]
    ) -> pd.DataFrame:
        """
        Converts given columns to numeric, filling invalid entries with 0.
        """
        for col in columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        return df

    def add_image_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Loads image features from a file, reduces dimensionality using PCA, and merges into dataframe.
        """
        features = FileManager.load_from_pkl(self.image_features_path)

        averaged_features = {
            pid: np.mean(sections, axis=0)
            for pid, sections in features.items()
            if len(sections) == 3 and all(len(s) == 1024 for s in sections)
        }

        features_df = pd.DataFrame.from_dict(averaged_features, orient="index")
        features_df.index.name = "Photo id"

        if not hasattr(self, "pca"):
            self.pca = PCA(n_components=self.use_images)
        reduced_features = self.pca.fit_transform(features_df)
        reduced_features_df = pd.DataFrame(
            reduced_features,
            index=features_df.index,
            columns=[f"Image_Feature_{i+1}" for i in range(self.use_images)],
        )

        return df.merge(
            reduced_features_df, left_on="Photo id", right_index=True, how="left"
        )

    def generate_combined_embeddings(self, df: pd.DataFrame) -> np.ndarray:
        """
        Generates combined text + numeric embeddings from the dataframe.
        """
        text_columns = self.TEXT_COLUMNS + self.additional_text_columns
        numeric_columns = self.NUMERIC_COLUMNS + self.additional_numeric_columns

        if self.use_images:
            numeric_columns += [f"Image_Feature_{i+1}" for i in range(self.use_images)]

        if self.encode_per_column:
            embeddings = np.hstack(
                [
                    self.model.encode(df[col].astype(str).tolist()).astype("float32")
                    for col in text_columns
                ]
            )
        else:
            descriptions = (
                df[text_columns]
                .astype(str)
                .apply(
                    lambda row: ", ".join(f"{col}: {val}" for col, val in row.items()),
                    axis=1,
                )
            )
            embeddings = self.model.encode(descriptions.tolist()).astype("float32")

        if self.use_separate_numeric_features:
            if not hasattr(self, "scaler"):
                self.scaler = StandardScaler()
            numeric_data = self.scaler.fit_transform(df[numeric_columns])
            return np.hstack((embeddings, numeric_data))

        return embeddings

    def preprocess_data(self, df: pd.DataFrame, is_training: bool) -> pd.DataFrame:
        """
        Applies the full preprocessing pipeline: filling missing data, removing outliers,
        adding images, and enriching with additional data.

        Args:
            df (pd.DataFrame): The dataset to preprocess.
            is_training (bool): Whether the preprocessing is for training or prediction.
        """
        # Add image features if enabled
        if self.use_images:
            df = self.add_image_features(df)

        # Reset the additional columns lists
        self.additional_text_columns = []
        self.additional_numeric_columns = []

        # Add additional data dynamically
        if self.use_artfacts:
            df, art_text, art_numeric = PropertyModifier.add_additional_data(
                df=df,
                df_key_column="Artist name",
                csv_path=self.artist_info_path,
                csv_key_column="Name",
                columns_to_exclude=["Birth Year", "Death Year"],
                # columns_to_include=["Nationality"],
                use_fuzzy_matching=True,
                use_partial_matching=True,
            )
            self.additional_text_columns += art_text
            self.additional_numeric_columns += art_numeric

        if self.use_count:
            df, count_text, count_numeric = PropertyModifier.add_additional_data(
                df=df,
                df_key_column="Artist name",
                csv_path=self.artist_count_path,
                csv_key_column="name",
                columns_to_exclude=["path"],
            )
            self.additional_text_columns += count_text
            self.additional_numeric_columns += count_numeric

            # Additional logic to calculate artwork counts for Lithuanian artists as they're missing in CSV.
            if "count" in df.columns:
                artist_counts = df["Artist name"].value_counts()
                missing_mask = df["count"].isna()
                df.loc[missing_mask, "count"] = df.loc[missing_mask, "Artist name"].map(
                    artist_counts
                )

        if self.use_synthetic:
            df, synth_text, synth_numeric = PropertyModifier.add_additional_data(
                df=df,
                df_key_column="Artist name",
                csv_path=self.synthetic_paths[0],
                csv_key_column="Artist name",
                columns_to_exclude=["Score Explanations"],
                use_fuzzy_matching=True,
                use_partial_matching=True,
            )
            self.additional_text_columns += synth_text
            self.additional_numeric_columns += synth_numeric

            df, synth_text2, synth_numeric2 = PropertyModifier.add_additional_data(
                df=df,
                df_key_column="Auction House",
                csv_path=self.synthetic_paths[1],
                csv_key_column="Auction House",
                columns_to_exclude=["Score Explanations"],
                use_fuzzy_matching=True,
                use_partial_matching=True,
            )
            self.additional_text_columns += synth_text2
            self.additional_numeric_columns += synth_numeric2

        if self.use_artsy:
            df, artsy_text, artsy_numeric = PropertyModifier.add_additional_data(
                df=df,
                df_key_column="Artist name",
                csv_path=self.artsy_path,
                csv_key_column="Name",
            )
            self.additional_text_columns += artsy_text
            self.additional_numeric_columns += artsy_numeric

        # TODO: Move hardcoded values
        if self.use_count:
            # Additional filtering to keep relevant artists only (with no less than `min_artwork_count`)
            df = process_keep_relevant(df, min_artwork_count=None, verbose=False)

        # Fill missing values with "Unknown" or -1 based on the column type
        df = ensure_data_filled_and_correct(df)

        # Hot-encode provided columns (needs to be after filling missing data)
        if self.hot_encode_columns:
            for col in self.hot_encode_columns:
                if col in df.columns:
                    hot_encoded = df[col].str.get_dummies()
                    hot_encoded.columns = [f"{col}_{c}" for c in hot_encoded.columns]
                    df = pd.concat([df, hot_encoded], axis=1)
                    self.additional_numeric_columns += hot_encoded.columns.tolist()
                    if col in self.TEXT_COLUMNS:
                        self.TEXT_COLUMNS.remove(col)
                else:
                    print(f"Warning: Column '{col}' not found in DataFrame.")

        if is_training:
            df = process_keep_relevant(df, max_missing_percent=self.max_missing_percent)
        return df

    def calculate_midpoints(self, df: pd.DataFrame) -> pd.Series:
        midpoints = (df["Estimated Minimum Price"] + df["Estimated Maximum Price"]) / 2
        midpoints = midpoints.where(
            (df["Estimated Minimum Price"] > 0) & (df["Estimated Maximum Price"] > 0),
            df["Sold Price"],
        )
        return midpoints

    def evaluate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        df: pd.DataFrame,
        estimate_error_margin: float = 0.05,
    ) -> Dict[str, float]:
        """
        Calculate and print evaluation metrics, including custom metrics.

        Args:
            y_true (np.ndarray): Ground truth values.
            y_pred (np.ndarray): Predicted values.
            df (pd.DataFrame): DataFrame containing "Estimated Minimum Price" and "Estimated Maximum Price".

        Returns:
            Dict[str, float]: Dictionary of evaluation metrics.
        """
        metrics = {
            "MAE": mean_absolute_error(y_true, y_pred),
            "MSE": mean_squared_error(y_true, y_pred),
            "MAPE": mean_absolute_percentage_error(y_true, y_pred),
            "R2": r2_score(y_true, y_pred),
        }

        # Filter out rows where "Estimated Minimum Price" or "Estimated Maximum Price" is 0
        if (
            "Estimated Minimum Price" in df.columns
            and "Estimated Maximum Price" in df.columns
        ):
            valid_estimates = (df["Estimated Minimum Price"] > 0) & (
                df["Estimated Maximum Price"] > 0
            )
            filtered_df = df[valid_estimates].copy()
            filtered_y_pred = y_pred[valid_estimates]
            filtered_y_true = y_true[valid_estimates]

            if not filtered_df.empty:
                estimated_min = filtered_df["Estimated Minimum Price"]
                estimated_max = filtered_df["Estimated Maximum Price"]

                # Calculate "Within Estimates" for actual and predicted prices
                within_estimates_actual = (filtered_y_true >= estimated_min) & (
                    filtered_y_true <= estimated_max
                )
                within_estimates_predicted = (filtered_y_pred >= estimated_min) & (
                    filtered_y_pred <= estimated_max
                )

                # Add "Within Estimates" as columns to the DataFrame
                filtered_df.loc[:, "Within Estimates Actual"] = within_estimates_actual
                filtered_df.loc[:, "Within Estimates Predicted"] = (
                    within_estimates_predicted
                )

                # Add price ranges
                bins = [0, 100, 1000, 2000, 5000]
                labels = ["Low", "Medium", "High", "Very High"]
                filtered_df.loc[:, "Price Range"] = pd.cut(
                    filtered_y_true,
                    bins=bins,
                    labels=labels,
                )

                # Calculate percentages for each price range
                actual_within_percent = (
                    filtered_df.groupby("Price Range", observed=True)[
                        "Within Estimates Actual"
                    ].mean()
                    * 100
                )
                predicted_within_percent = (
                    filtered_df.groupby("Price Range", observed=True)[
                        "Within Estimates Predicted"
                    ].mean()
                    * 100
                )

                # Calculate deviations for actual and predicted prices
                actual_deviation = np.where(
                    filtered_y_true < estimated_min,
                    estimated_min - filtered_y_true,
                    np.where(
                        filtered_y_true > estimated_max,
                        filtered_y_true - estimated_max,
                        0,
                    ),
                )
                predicted_deviation = np.where(
                    filtered_y_pred < estimated_min,
                    estimated_min - filtered_y_pred,
                    np.where(
                        filtered_y_pred > estimated_max,
                        filtered_y_pred - estimated_max,
                        0,
                    ),
                )

                # Add deviations to the DataFrame
                filtered_df.loc[:, "Actual Deviation"] = actual_deviation
                filtered_df.loc[:, "Predicted Deviation"] = predicted_deviation

                # Group by price range and calculate mean deviations
                mean_deviation = filtered_df.groupby("Price Range", observed=True)[
                    ["Actual Deviation", "Predicted Deviation"]
                ].mean()

                # Expand the range by the error margin
                expanded_min = filtered_df["Estimated Minimum Price"] * (
                    1 - estimate_error_margin
                )
                expanded_max = filtered_df["Estimated Maximum Price"] * (
                    1 + estimate_error_margin
                )

                # Calculate custom metrics
                within_estimates = (filtered_y_pred >= expanded_min) & (
                    filtered_y_pred <= expanded_max
                )
                metrics["Within_Estimates_Percentage"] = within_estimates.mean() * 100

                above_max = filtered_y_pred > expanded_max
                metrics["Above_Max_Percentage"] = above_max.mean() * 100

                below_min = filtered_y_pred < expanded_min
                metrics["Below_Min_Percentage"] = below_min.mean() * 100

                # Calculate percentage error for predictions outside the range
                outside_range = ~within_estimates
                percentage_error = np.zeros_like(filtered_y_pred, dtype=float)

                # For predictions below the minimum range
                percentage_error[filtered_y_pred < expanded_min] = (
                    (
                        expanded_min[filtered_y_pred < expanded_min]
                        - filtered_y_pred[filtered_y_pred < expanded_min]
                    )
                    / expanded_min[filtered_y_pred < expanded_min]
                ) * 100

                # For predictions above the maximum range
                percentage_error[filtered_y_pred > expanded_max] = (
                    (
                        filtered_y_pred[filtered_y_pred > expanded_max]
                        - expanded_max[filtered_y_pred > expanded_max]
                    )
                    / expanded_max[filtered_y_pred > expanded_max]
                ) * 100

                # Average percentage error for predictions outside the range
                metrics["Average_Percentage_Error_Outside_Range"] = (
                    percentage_error[outside_range].mean()
                    if outside_range.any()
                    else 0.0
                )

                # Calculate "Within Estimates with 0% Error" for predictions
                within_exact_estimates = (
                    filtered_y_pred >= filtered_df["Estimated Minimum Price"]
                ) & (filtered_y_pred <= filtered_df["Estimated Maximum Price"])
                metrics["Within_Exact_Estimates_Percentage"] = (
                    within_exact_estimates.mean() * 100
                )

                # Calculate "Within Estimates with 0% Error" for actual sold prices
                within_exact_sold_prices = (
                    filtered_y_true >= filtered_df["Estimated Minimum Price"]
                ) & (filtered_y_true <= filtered_df["Estimated Maximum Price"])
                metrics["Sold_Price_Within_Exact_Estimates_Percentage"] = (
                    within_exact_sold_prices.mean() * 100
                )

                # Print custom metrics
                print(
                    f"Within Estimates Percentage (with {estimate_error_margin*100:.1f}% margin): {metrics['Within_Estimates_Percentage']:.2f}%"
                )
                print(
                    f"Within Exact Estimates Percentage: {metrics['Within_Exact_Estimates_Percentage']:.2f}% "
                    f"({metrics['Sold_Price_Within_Exact_Estimates_Percentage']:.2f}% for actual Sold Price)"
                )
                print(f"Above Max Percentage: {metrics['Above_Max_Percentage']:.2f}%")
                print(f"Below Min Percentage: {metrics['Below_Min_Percentage']:.2f}%")
                print(
                    f"Average Percentage Error Outside Range: {metrics['Average_Percentage_Error_Outside_Range']:.2f}%"
                )
            else:
                print("Warning: No valid rows for custom metric calculation.")
        else:
            print(
                "Warning: 'Estimated Minimum Price' or 'Estimated Maximum Price' columns not found in DataFrame."
            )

        # Print other metrics
        print(f"MAE: {metrics['MAE']}")
        print(f"MSE: {metrics['MSE']}")
        print(f"MAPE: {metrics['MAPE']}")
        print(f"R2: {metrics['R2']}")

        self.create_evaluation_plots(y_true, y_pred, df, metrics, tolerance=0.2)

        return metrics

    def log_results(
        self,
        metrics: Dict[str, float],
        predictor_info: Dict[str, Any],
        regressor_names: List[str],
        df_size: tuple,
        nn_info: Dict[str, Any] = None,
        log_dir: str = "logs",
    ) -> None:
        """
        Save metrics, predictor parameters, and regressor information to a CSV log file.

        Args:
            metrics: Dictionary of evaluation metrics (MAE, MSE, MAPE, R2)
            predictor_info: Dictionary of predictor parameters
            regressor_names: List of regressor names used
            df_size: Tuple containing (rows, columns) of the preprocessed DataFrame
            nn_info: Dictionary with neural network specific information (model_class, loss_function, hidden_units)
            log_dir: Directory to save log files
        """
        os.makedirs(log_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(log_dir, f"training_log_{timestamp}.csv")

        log_data = {
            "Regressors": ";".join(regressor_names),
            "Data Rows": df_size[0],
            "Used Data Columns count": df_size[1],
        }

        if nn_info:
            log_data["ModelClass"] = nn_info.get("model_class", "")
            log_data["LossFunction"] = nn_info.get("loss_function", "")
            log_data["HiddenUnits"] = nn_info.get("hidden_units", "")
        else:
            log_data["ModelClass"] = ""
            log_data["LossFunction"] = ""
            log_data["HiddenUnits"] = ""

        log_data.update(metrics)

        for key, value in predictor_info.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                log_data[f"predictor_{key}"] = value

        pd.DataFrame([log_data]).to_csv(log_path, index=False)
        print(f"Training results logged to {log_path}")

    def create_evaluation_plots(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        df: pd.DataFrame,
        metrics: Dict[str, float],
        tolerance: float = 0.1,
    ) -> None:
        """
        Creates a single plot containing multiple evaluation graphs:
        1. Scatter plot of actual vs. predicted prices.
        2. Error distribution by price range.
        3. Cumulative error plot.
        4. Price range accuracy bar chart.
        5. Within Estimates Plot (Predicted vs. Actual vs. Estimated Ranges).

        Args:
            y_true (np.ndarray): Ground truth values.
            y_pred (np.ndarray): Predicted values.
            df (pd.DataFrame): DataFrame containing the actual prices and other metadata.
            metrics (Dict[str, float]): Precomputed metrics dictionary.
            tolerance (float): Tolerance for the price range accuracy bar chart.
        """
        # Prepare data
        bins = [0, 100, 1000, 2000, 5000]
        labels = [
            "Low",
            "Medium",
            "High",
            "Very High",
        ]

        df_plot = pd.DataFrame({"Actual": y_true, "Predicted": y_pred})
        df_plot["Price Range"] = pd.cut(
            df_plot["Actual"],
            bins=bins,
            labels=labels,
        )

        # Create subplots (3 horizontal, 2 vertical)
        fig, axes = plt.subplots(3, 2, figsize=(18, 16))
        fig.suptitle("Evaluation Plots", fontsize=18)

        # Add a legend at the top for price ranges using a dummy plot
        handles = [
            plt.Line2D(
                [0], [0], color="none", label=f"{label}: {bins[i]} - {bins[i + 1]}"
            )
            for i, label in enumerate(labels)
        ]
        fig.legend(
            handles=handles,
            loc="upper center",
            fontsize=10,
            title="Price Ranges",
            ncol=len(labels),
            frameon=True,
            bbox_to_anchor=(0.5, 0.95),
        )

        # 1. Scatter Plot of Actual vs. Predicted Prices
        axes[0, 0].scatter(
            df_plot["Actual"],
            df_plot["Predicted"],
            alpha=0.6,
            c=abs(y_true - y_pred),  # Use precomputed absolute error
            cmap="coolwarm",
        )
        axes[0, 0].plot(
            [df_plot["Actual"].min(), df_plot["Actual"].max()],
            [df_plot["Actual"].min(), df_plot["Actual"].max()],
            color="green",
            linestyle="--",
        )
        axes[0, 0].set_title("Actual vs. Predicted Prices")
        axes[0, 0].set_xlabel("Actual Price")
        axes[0, 0].set_ylabel("Predicted Price")
        axes[0, 0].grid()

        # 2. Error Distribution by Price Range
        sns.boxplot(
            x="Price Range", y=abs(y_true - y_pred), data=df_plot, ax=axes[0, 1]
        )
        axes[0, 1].set_title("Error Distribution by Price Range")
        axes[0, 1].set_xlabel("Price Range")
        axes[0, 1].set_ylabel("Absolute Error")

        # 3. Cumulative Error Plot
        thresholds = range(0, int(abs(y_true - y_pred).max()), 50)
        cumulative_percentage = [
            sum(abs(y_true - y_pred) <= t) / len(y_true) * 100 for t in thresholds
        ]
        axes[1, 0].plot(thresholds, cumulative_percentage, marker="o")
        axes[1, 0].set_title("Cumulative Error Plot")
        axes[1, 0].set_xlabel("Error Threshold")
        axes[1, 0].set_ylabel("Cumulative Percentage of Predictions")
        axes[1, 0].grid()

        # 4. Price Range Accuracy Bar Chart with Multiple Thresholds
        thresholds = [0.05, 0.1, 0.2, 0.5]  # Define thresholds (5%, 10%, 20%, 50%)
        accuracy_by_range = {}

        # Calculate accuracy for each threshold
        for threshold in thresholds:
            df_plot[f"Within Tolerance {int(threshold * 100)}%"] = (
                abs(y_true - y_pred) <= threshold * y_true
            )
            accuracy_by_range[threshold] = (
                df_plot.groupby("Price Range", observed=True)[
                    f"Within Tolerance {int(threshold * 100)}%"
                ].mean()
                * 100
            )

        # Plot grouped bars
        x = np.arange(len(labels))  # X positions for groups (price ranges)
        bar_width = 0.2  # Width of each bar

        for i, threshold in enumerate(thresholds):
            axes[1, 1].bar(
                x
                + (i - len(thresholds) / 2) * bar_width
                + bar_width / 2,  # Offset bars
                accuracy_by_range[threshold].values,
                bar_width,
                label=f"{int(threshold * 100)}% Tolerance",
            )

        # Configure the plot
        axes[1, 1].set_title("Accuracy by Price Range for Multiple Tolerances")
        axes[1, 1].set_xlabel("Price Range")
        axes[1, 1].set_ylabel("Accuracy (%)")
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(labels)
        axes[1, 1].legend(title="Tolerance")
        axes[1, 1].grid()

        # 6. Within Estimates Plot (Predicted vs. Actual vs. Estimated Ranges)
        valid_estimates = (df["Estimated Minimum Price"] > 0) & (
            df["Estimated Maximum Price"] > 0
        )
        filtered_df = df[valid_estimates].copy()
        filtered_y_true = y_true[valid_estimates]
        filtered_y_pred = y_pred[valid_estimates]

        estimated_min = filtered_df["Estimated Minimum Price"]
        estimated_max = filtered_df["Estimated Maximum Price"]

        within_estimates_actual = (filtered_y_true >= estimated_min) & (
            filtered_y_true <= estimated_max
        )
        within_estimates_predicted = (filtered_y_pred >= estimated_min) & (
            filtered_y_pred <= estimated_max
        )

        filtered_df.loc[:, "Within Estimates Actual"] = within_estimates_actual
        filtered_df.loc[:, "Within Estimates Predicted"] = within_estimates_predicted

        filtered_df.loc[:, "Price Range"] = pd.cut(
            filtered_y_true,
            bins=bins,
            labels=labels,
        )

        actual_within_percent = (
            filtered_df.groupby("Price Range", observed=True)[
                "Within Estimates Actual"
            ].mean()
            * 100
        )
        predicted_within_percent = (
            filtered_df.groupby("Price Range", observed=True)[
                "Within Estimates Predicted"
            ].mean()
            * 100
        )

        actual_deviation = np.where(
            filtered_y_true < estimated_min,
            estimated_min - filtered_y_true,
            np.where(
                filtered_y_true > estimated_max, filtered_y_true - estimated_max, 0
            ),
        )
        predicted_deviation = np.where(
            filtered_y_pred < estimated_min,
            estimated_min - filtered_y_pred,
            np.where(
                filtered_y_pred > estimated_max, filtered_y_pred - estimated_max, 0
            ),
        )

        filtered_df.loc[:, "Actual Deviation"] = actual_deviation
        filtered_df.loc[:, "Predicted Deviation"] = predicted_deviation

        mean_deviation = filtered_df.groupby("Price Range", observed=True)[
            ["Actual Deviation", "Predicted Deviation"]
        ].mean()

        x = np.arange(len(actual_within_percent))
        bar_width = 0.35
        axes[2, 1].bar(
            x - bar_width / 2,
            actual_within_percent.values,
            bar_width,
            color="blue",
            alpha=0.6,
            label="Actual Sold Price",
        )
        axes[2, 1].bar(
            x + bar_width / 2,
            predicted_within_percent.values,
            bar_width,
            color="orange",
            alpha=0.6,
            label="Predicted Price",
        )
        axes[2, 1].set_title("Percentage Within Estimates by Price Range")
        axes[2, 1].set_xlabel("Price Range")
        axes[2, 1].set_ylabel("Percentage Within Estimates (%)")
        axes[2, 1].set_xticks(x)
        axes[2, 1].set_xticklabels(actual_within_percent.index)
        axes[2, 1].legend()
        axes[2, 1].grid()

        actual_deviation = np.where(
            filtered_y_true < estimated_min,
            estimated_min - filtered_y_true,
            np.where(
                filtered_y_true > estimated_max, filtered_y_true - estimated_max, 0
            ),
        )
        predicted_deviation = np.where(
            filtered_y_pred < estimated_min,
            estimated_min - filtered_y_pred,
            np.where(
                filtered_y_pred > estimated_max, filtered_y_pred - estimated_max, 0
            ),
        )

        filtered_df.loc[:, "Actual Deviation"] = actual_deviation
        filtered_df.loc[:, "Predicted Deviation"] = predicted_deviation

        mean_deviation = filtered_df.groupby("Price Range", observed=True)[
            ["Actual Deviation", "Predicted Deviation"]
        ].mean()

        # 5. Plot Heatmap of Deviations
        sns.heatmap(
            mean_deviation.T,
            annot=True,
            cmap="coolwarm",
            fmt=".2f",
            ax=axes[2, 0],
        )
        axes[2, 0].set_title("Mean Deviation from Estimated Range by Price Range")
        axes[2, 0].set_xlabel("Price Range")
        axes[2, 0].set_ylabel("Deviation Type")

        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.92])  # Leave space for the legend
        plt.show()
