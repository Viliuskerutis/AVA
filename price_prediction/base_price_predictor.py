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

    # NUMERIC_COLUMNS = [
    #     "Max Sold Price",
    #     "Artist Birth Year",
    #     "Min Sold Price",
    #     "Creation Year",
    #     "Years from auction till now",
    #     "Width",
    #     "Height",
    #     "Average Sold Price",
    #     "Area",
    # ]

    # TEXT_COLUMNS = [
    #     "Auction House",
    #     "Artist name",
    #     "Auction Country",
    #     "Surface",
    #     "Materials",
    # ]

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
            self.pca = PCA(n_components=self.use_images, random_state=42)
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
            numeric_columns = sorted(numeric_columns)
            if not hasattr(self, "scaler"):
                self.scaler = StandardScaler()
                self.scaler.fit(df[numeric_columns])
            numeric_data = self.scaler.transform(df[numeric_columns])
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
        if is_training and hasattr(self, "preprocessed_df"):
            print("Using cached preprocessed DataFrame.")
            return self.preprocessed_df

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
                # columns_to_include=[
                #     "Biennials",
                #     "Art Fairs",
                #     "Solo Exhibitions",
                #     "Exhibitions Most With 1 No",
                #     "Verified Exhibitions",
                #     "Exhibitions Most At 2 No",
                #     "Exhibitions Top Country 1 No",
                #     "Description",
                #     "Exhibitions Most With 2",
                #     "Notable Exhibitions At 1",
                #     "Exhibitions Top Country 1",
                #     "Exhibitions Most At 1",
                # ],
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
                PropertyModifier.save_missing_data_to_csv(
                    df=df[missing_mask],
                    csv_path=self.artist_count_path,
                    df_key_column="Artist name",
                    csv_key_column="name",
                    missing_column="count",
                    verbose=True,
                )

        if self.use_synthetic:
            df, synth_text, synth_numeric = PropertyModifier.add_additional_data(
                df=df,
                df_key_column="Artist name",
                csv_path=self.synthetic_paths[0],
                csv_key_column="Artist name",
                columns_to_exclude=["Score Explanations"],
                # columns_to_include=[
                #     "Auction Record Price",
                #     "Artistic Styles",
                #     "Primary Market",
                # ],
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
                # columns_to_include=[
                #     "Repeat Buyer Percentage",
                #     "Specialization Score",
                #     "Prestige Rank",
                #     "Top Sale Price Record",
                #     "Marketing Power Rank",
                #     "Liquidity Score",
                # ],
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
            self.preprocessed_df = df
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

        # Set x-ticks more frequently
        axes[1, 0].set_xticks(range(0, int(abs(y_true - y_pred).max()) + 1, 200))
        axes[1, 0].tick_params(axis="x", rotation=60)
        axes[1, 0].set_yticks(range(0, 101, 10))  # Every 10% for y-axis

        # Add grid for both x and y ticks
        axes[1, 0].grid(which="both", linestyle="--", linewidth=0.5)

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

        # Align accuracy_by_range with all price range labels and fill missing values with 0
        for threshold in thresholds:
            accuracy_by_range[threshold] = accuracy_by_range[threshold].reindex(
                labels, fill_value=0
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

        filtered_df.loc[:, "Actual"] = actual_deviation
        filtered_df.loc[:, "Predicted"] = predicted_deviation

        mean_deviation = filtered_df.groupby("Price Range", observed=True)[
            ["Actual", "Predicted"]
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

        plt.tight_layout(rect=[0, 0.05, 1, 0.92])  # Leave space for the legend
        plt.subplots_adjust(hspace=0.4)  # Increase the value to add more space

        plt.show(block=False)

    def evaluate_custom_estimates(
        self, results_df: pd.DataFrame
    ) -> Dict[str, Dict[str, float]]:
        custom_estimate_tables = [
            [
                (100, 0.2),
                (200, 0.2),
                (500, 0.15),
                (3000, 0.1),
                (10000, 0.1),
                (200000, 0.07),
            ],
            [
                (100, 0.15),
                (200, 0.15),
                (500, 0.1),
                (3000, 0.05),
                (10000, 0.05),
                (200000, 0.03),
            ],
            [
                (100, 0.25),
                (200, 0.25),
                (500, 0.2),
                (3000, 0.15),
                (10000, 0.15),
                (200000, 0.1),
            ],
        ]

        results = {}

        for i, custom_estimate_table in enumerate(custom_estimate_tables):

            def calculate_custom_estimates(predicted_price):
                for price, percentage in custom_estimate_table:
                    if predicted_price <= price:
                        added_value = predicted_price * percentage
                        return (
                            predicted_price - added_value,
                            predicted_price + added_value,
                        )
                percentage = custom_estimate_table[-1][1]
                added_value = predicted_price * percentage
                return (
                    predicted_price - added_value,
                    predicted_price + added_value,
                )

            filtered_df = results_df[
                (results_df["Estimated Minimum Price"] > 0)
                & (results_df["Estimated Maximum Price"] > 0)
            ].copy()

            filtered_df[["Custom Min Estimate", "Custom Max Estimate"]] = filtered_df[
                "Predicted"
            ].apply(lambda x: pd.Series(calculate_custom_estimates(x)))

            filtered_df["Custom Min Within Professional"] = (
                filtered_df["Custom Min Estimate"]
                >= filtered_df["Estimated Minimum Price"]
            ) & (
                filtered_df["Custom Min Estimate"]
                <= filtered_df["Estimated Maximum Price"]
            )
            filtered_df["Custom Max Within Professional"] = (
                filtered_df["Custom Max Estimate"]
                >= filtered_df["Estimated Minimum Price"]
            ) & (
                filtered_df["Custom Max Estimate"]
                <= filtered_df["Estimated Maximum Price"]
            )
            filtered_df["Custom Within Professional"] = (
                filtered_df["Custom Min Within Professional"]
                & filtered_df["Custom Max Within Professional"]
            )

            filtered_df["Actual Within Professional Estimate"] = (
                filtered_df["Actual"] >= filtered_df["Estimated Minimum Price"]
            ) & (filtered_df["Actual"] <= filtered_df["Estimated Maximum Price"])
            filtered_df["Actual Within Custom Estimate"] = (
                filtered_df["Actual"] >= filtered_df["Custom Min Estimate"]
            ) & (filtered_df["Actual"] <= filtered_df["Custom Max Estimate"])
            filtered_df["Predicted Within Professional Estimate"] = (
                filtered_df["Predicted"] >= filtered_df["Estimated Minimum Price"]
            ) & (filtered_df["Predicted"] <= filtered_df["Estimated Maximum Price"])

            filtered_df["Custom-Professional Intercept"] = (
                filtered_df[["Custom Max Estimate", "Estimated Maximum Price"]].min(
                    axis=1
                )
                - filtered_df[["Custom Min Estimate", "Estimated Minimum Price"]].max(
                    axis=1
                )
            ).clip(lower=0)

            filtered_df["Custom-Professional Intercept Percentage"] = filtered_df[
                "Custom-Professional Intercept"
            ] / (
                filtered_df["Estimated Maximum Price"]
                - filtered_df["Estimated Minimum Price"]
            )

            filtered_df["Custom Min vs Professional Min Diff"] = (
                filtered_df["Custom Min Estimate"]
                - filtered_df["Estimated Minimum Price"]
            ) / filtered_df["Estimated Minimum Price"]
            filtered_df["Custom Max vs Professional Max Diff"] = (
                filtered_df["Custom Max Estimate"]
                - filtered_df["Estimated Maximum Price"]
            ) / filtered_df["Estimated Maximum Price"]
            filtered_df["Custom Min Absolute Difference"] = abs(
                filtered_df["Custom Min Estimate"]
                - filtered_df["Estimated Minimum Price"]
            )
            filtered_df["Custom Max Absolute Difference"] = abs(
                filtered_df["Custom Max Estimate"]
                - filtered_df["Estimated Maximum Price"]
            )

            filtered_df["Custom Min Absolute Percentage Difference"] = (
                abs(
                    (
                        filtered_df["Custom Min Estimate"]
                        - filtered_df["Estimated Minimum Price"]
                    )
                    / filtered_df["Estimated Minimum Price"]
                )
                * 100
            )

            filtered_df["Custom Max Absolute Percentage Difference"] = (
                abs(
                    (
                        filtered_df["Custom Max Estimate"]
                        - filtered_df["Estimated Maximum Price"]
                    )
                    / filtered_df["Estimated Maximum Price"]
                )
                * 100
            )

            alignment_summary = {
                "Custom Within Professional (%)": filtered_df[
                    "Custom Within Professional"
                ].mean()
                * 100,
                "Custom Min Within Professional (%)": filtered_df[
                    "Custom Min Within Professional"
                ].mean()
                * 100,
                "Custom Max Within Professional (%)": filtered_df[
                    "Custom Max Within Professional"
                ].mean()
                * 100,
                "Custom-Professional Intercept (%)": filtered_df[
                    "Custom-Professional Intercept Percentage"
                ].mean(),
                "Actual Within Professional Estimate (%)": filtered_df[
                    "Actual Within Professional Estimate"
                ].mean()
                * 100,
                "Actual Within Custom Estimate (%)": filtered_df[
                    "Actual Within Custom Estimate"
                ].mean()
                * 100,
                "Predicted Within Professional Estimate (%)": filtered_df[
                    "Predicted Within Professional Estimate"
                ].mean()
                * 100,
                "Custom Min vs Professional Min Diff (%)": filtered_df[
                    "Custom Min vs Professional Min Diff"
                ].mean()
                * 100,
                "Custom Min Absolute Difference": filtered_df[
                    "Custom Min Absolute Difference"
                ].mean(),
                "Custom Max vs Professional Max Diff (%)": filtered_df[
                    "Custom Max vs Professional Max Diff"
                ].mean()
                * 100,
                "Custom Max Absolute Difference": filtered_df[
                    "Custom Max Absolute Difference"
                ].mean(),
                "Custom Min Absolute Percentage Difference (%)": filtered_df[
                    "Custom Min Absolute Percentage Difference"
                ].mean(),
                "Custom Max Absolute Percentage Difference (%)": filtered_df[
                    "Custom Max Absolute Percentage Difference"
                ].mean(),
            }

            results[f"Table {i + 1}"] = alignment_summary

            print(f"Table {i + 1} - Custom Estimates Summary:")
            for metric, value in alignment_summary.items():
                if (
                    metric != "Custom Min Absolute Difference"
                    and metric != "Custom Max Absolute Difference"
                ):
                    print(f"  {metric}: {value:.2f}%")
                else:
                    print(f"  {metric}: {value:.2f}")

            # self.plot_estimate_metrics(filtered_df, table_name=f"Table {i + 1}")
            self.plot_custom_vs_professional_metrics(
                filtered_df, table_name=f"Table {i + 1}"
            )

        return results

    def plot_estimate_metrics(
        self,
        results_df: pd.DataFrame,
        table_name: str,
    ) -> None:
        bins = [0, 100, 1000, 2000, 5000]
        labels = ["Low", "Medium", "High", "Very High"]

        results_df["Price Range"] = pd.cut(
            results_df["Predicted"], bins=bins, labels=labels
        )

        # Create a figure with subplots
        fig, axes = plt.subplots(4, 2, figsize=(18, 20))
        fig.suptitle(f"Interception Metrics for {table_name}", fontsize=16)

        all_counts = results_df["Price Range"].value_counts(sort=False)

        # Add a legend at the top for price ranges
        handles = [
            plt.Line2D(
                [0],
                [0],
                color="none",
                label=f"{label} {bins[i]}-{bins[i + 1]} ({all_counts[label]})",
            )
            for i, label in enumerate(labels)
        ]
        fig.legend(
            handles=handles,
            loc="upper center",
            fontsize=10,
            title="Price Ranges (Element count)",
            ncol=len(labels),
            frameon=True,
            bbox_to_anchor=(0.5, 0.95),
        )

        # 1. Bar Chart: Percentage of "Actual" Within Custom Range by Price Range
        actual_within_custom = (
            results_df.groupby("Price Range", observed=True)[
                "Actual Within Custom Estimate"
            ].mean()
            * 100
        )

        axes[0, 0].bar(
            actual_within_custom.index,
            actual_within_custom,
            color="lightcoral",
            edgecolor="black",
        )
        axes[0, 0].set_title(
            "Percentage of 'Actual' Within Custom Range by Price Range"
        )
        axes[0, 0].set_xlabel("Price Range")
        axes[0, 0].set_ylabel("Percentage (%)")
        axes[0, 0].grid(axis="y", linestyle="--", alpha=0.7)

        # 2. Heatmap: Actual Deviation from Custom Range by Price Range
        actual_deviation_from_custom = results_df.copy()

        actual_deviation_from_custom["Deviation Below Custom"] = np.where(
            actual_deviation_from_custom["Actual"]
            < actual_deviation_from_custom["Custom Min Estimate"],
            actual_deviation_from_custom["Custom Min Estimate"]
            - actual_deviation_from_custom["Actual"],
            0,
        )

        actual_deviation_from_custom["Deviation Above Custom"] = np.where(
            actual_deviation_from_custom["Actual"]
            > actual_deviation_from_custom["Custom Max Estimate"],
            actual_deviation_from_custom["Actual"]
            - actual_deviation_from_custom["Custom Max Estimate"],
            0,
        )

        mean_deviation_custom = actual_deviation_from_custom.groupby(
            "Price Range", observed=True
        )[["Deviation Below Custom", "Deviation Above Custom"]].mean()

        sns.heatmap(
            mean_deviation_custom.T,
            annot=True,
            cmap="coolwarm",
            fmt=".2f",
            ax=axes[0, 1],
        )
        axes[0, 1].set_title(
            "Mean Deviation of 'Actual' from Custom Range by Price Range"
        )
        axes[0, 1].set_xlabel("Price Range")
        axes[0, 1].set_ylabel("Deviation Type")

        # 3. Bar Chart: Average Interception Percentage by Price Range
        avg_interception = results_df.groupby("Price Range", observed=True)[
            "Custom-Professional Intercept Percentage"
        ].mean()

        axes[1, 0].bar(
            avg_interception.index, avg_interception, color="skyblue", edgecolor="black"
        )
        axes[1, 0].set_title("Average Interception Percentage by Price Range")
        axes[1, 0].set_xlabel("Price Range")
        axes[1, 0].set_ylabel("Average Interception Percentage (%)")
        axes[1, 0].grid(axis="y", linestyle="--", alpha=0.7)

        # 4. Box Plot: Distribution of Interception Percentage by Price Range
        sns.boxplot(
            x="Price Range",
            y="Custom-Professional Intercept Percentage",
            data=results_df,
            ax=axes[1, 1],
        )
        axes[1, 1].set_title("Distribution of Interception Percentage by Price Range")
        axes[1, 1].set_xlabel("Price Range")
        axes[1, 1].set_ylabel("Interception Percentage (%)")
        axes[1, 1].grid(axis="y", linestyle="--", alpha=0.7)

        # 5. Stacked Bar Chart: Breakdown of Interception Cases by Interception Bins
        interception_bins = [0, 25, 50, 75, 100]
        interception_labels = ["0-25%", "25-50%", "50-75%", "75-100%"]

        results_df["Interception Bin"] = pd.cut(
            results_df["Custom-Professional Intercept Percentage"],
            bins=interception_bins,
            labels=interception_labels,
        )

        stacked_data = (
            results_df.groupby(["Price Range", "Interception Bin"], observed=True)
            .size()
            .unstack()
        )
        stacked_data = stacked_data.div(stacked_data.sum(axis=1), axis=0) * 100

        stacked_data.plot(kind="bar", stacked=True, colormap="viridis", ax=axes[2, 0])
        axes[2, 0].set_title("Breakdown of Interception Percentage by Price Range")
        axes[2, 0].set_xlabel("Price Range")
        axes[2, 0].set_ylabel("Percentage of Cases (%)")
        axes[2, 0].set_xticklabels(stacked_data.index, rotation=0, ha="center")
        axes[2, 0].grid(axis="y", linestyle="--", alpha=0.7)

        # Add legend inside the plot on the right
        axes[2, 0].legend(
            title="Interception",
            loc="center left",
            bbox_to_anchor=(
                1.0,
                0.5,
            ),  # Position the legend inside the plot on the right
            frameon=True,
        )

        # 6. Scatter Plot: Interception Percentage vs. Predicted Price
        sns.scatterplot(
            x="Predicted",
            y="Custom-Professional Intercept Percentage",
            hue="Price Range",
            data=results_df,
            palette="Set2",
            alpha=0.7,
            ax=axes[2, 1],
        )
        axes[2, 1].set_title("Interception Percentage vs. Predicted Price")
        axes[2, 1].set_xlabel("Predicted Price")
        axes[2, 1].set_ylabel("Interception Percentage (%)")
        axes[2, 1].grid(linestyle="--", alpha=0.7)

        # 7. Bar Chart: Custom Within Professional Percentage by Price Range
        custom_within_professional = (
            results_df.groupby("Price Range", observed=True)[
                "Custom Within Professional"
            ].mean()
            * 100
        )

        axes[3, 0].bar(
            custom_within_professional.index,
            custom_within_professional,
            color="lightgreen",
            edgecolor="black",
        )
        axes[3, 0].set_title("Custom Within Professional Percentage by Price Range")
        axes[3, 0].set_xlabel("Price Range")
        axes[3, 0].set_ylabel("Percentage (%)")
        axes[3, 0].grid(axis="y", linestyle="--", alpha=0.7)

        # 8. Line Plot: Custom Within Professional Percentage Across Predicted Prices
        results_df["Predicted Bin"] = pd.cut(
            results_df["Predicted"],
            bins=10,  # Divide predicted prices into 20 bins
        )

        line_data = (
            results_df.groupby("Predicted Bin", observed=True)[
                "Custom Within Professional"
            ].mean()
            * 100
        )

        bin_labels = [
            f"{int(interval.left)}-{int(interval.right)}"
            for interval in line_data.index.categories
        ]

        axes[3, 1].plot(
            range(len(line_data)),
            line_data.values,
            marker="o",
            linestyle="-",
            color="purple",
            alpha=0.8,
        )
        axes[3, 1].set_title(
            "Custom Within Professional Percentage Across Predicted Prices"
        )
        axes[3, 1].set_xlabel("Predicted Price Range")
        axes[3, 1].set_ylabel("Percentage (%)")
        axes[3, 1].set_xticks(range(len(bin_labels)))
        axes[3, 1].set_xticklabels(bin_labels, rotation=45, ha="right")
        axes[3, 1].grid(linestyle="--", alpha=0.7)

        # Adjust layout
        plt.tight_layout(rect=[0, 0.05, 1, 0.9])
        plt.subplots_adjust(hspace=0.6)
        plt.show(block=False)

    def plot_custom_vs_professional_metrics(
        self, results_df: pd.DataFrame, table_name: str
    ) -> None:
        """
        Creates a plot with 8 subplots:
        1. Bar chart showing the average percentage difference of Custom Min vs Professional Min by price range.
        2. Bar chart showing the average percentage difference of Custom Max vs Professional Max by price range.
        3. Bar chart showing the average absolute percentage difference of Custom Min by price range.
        4. Bar chart showing the average absolute percentage difference of Custom Max by price range.
        5. Scatter plot of Custom Min vs Professional Min with color gradient.
        6. Scatter plot of Custom Max vs Professional Max with color gradient.
        7. Box plot of Custom Min vs Professional Min differences by price range.
        8. Box plot of Custom Max vs Professional Max differences by price range.
        """
        bins = [0, 100, 1000, 2000, 5000]
        labels = ["Low", "Medium", "High", "Very High"]

        # Add price range column
        results_df["Price Range"] = pd.cut(
            results_df["Predicted"], bins=bins, labels=labels
        )

        # Group by price range and calculate the mean percentage differences
        avg_min_diff = (
            results_df.groupby("Price Range", observed=True)[
                "Custom Min vs Professional Min Diff"
            ].mean()
            * 100
        )
        avg_max_diff = (
            results_df.groupby("Price Range", observed=True)[
                "Custom Max vs Professional Max Diff"
            ].mean()
            * 100
        )
        avg_min_abs_percentage_diff = results_df.groupby("Price Range", observed=True)[
            "Custom Min Absolute Percentage Difference"
        ].mean()
        avg_max_abs_percentage_diff = results_df.groupby("Price Range", observed=True)[
            "Custom Max Absolute Percentage Difference"
        ].mean()

        fig, axes = plt.subplots(4, 2, figsize=(16, 24))
        fig.suptitle(f"Custom vs Professional Metrics - {table_name}", fontsize=16)

        # Plot 1: Custom Min vs Professional Min (%)
        axes[0, 0].bar(
            avg_min_diff.index,
            avg_min_diff,
            color="skyblue",
            edgecolor="black",
        )
        axes[0, 0].set_title("Custom Min vs Professional Min (%)")
        axes[0, 0].set_xlabel("Price Range")
        axes[0, 0].set_ylabel("Average Percentage Difference (%)")
        axes[0, 0].grid(axis="y", linestyle="--", alpha=0.7)

        # Plot 2: Custom Max vs Professional Max (%)
        axes[0, 1].bar(
            avg_max_diff.index,
            avg_max_diff,
            color="lightcoral",
            edgecolor="black",
        )
        axes[0, 1].set_title("Custom Max vs Professional Max (%)")
        axes[0, 1].set_xlabel("Price Range")
        axes[0, 1].set_ylabel("Average Percentage Difference (%)")
        axes[0, 1].grid(axis="y", linestyle="--", alpha=0.7)

        # Plot 3: Custom Min Absolute Percentage Difference
        axes[1, 0].bar(
            avg_min_abs_percentage_diff.index,
            avg_min_abs_percentage_diff,
            color="lightgreen",
            edgecolor="black",
        )
        axes[1, 0].set_title("Custom Min Absolute Percentage Difference")
        axes[1, 0].set_xlabel("Price Range")
        axes[1, 0].set_ylabel("Average Absolute Percentage Difference (%)")
        axes[1, 0].grid(axis="y", linestyle="--", alpha=0.7)

        # Plot 4: Custom Max Absolute Percentage Difference
        axes[1, 1].bar(
            avg_max_abs_percentage_diff.index,
            avg_max_abs_percentage_diff,
            color="gold",
            edgecolor="black",
        )
        axes[1, 1].set_title("Custom Max Absolute Percentage Difference")
        axes[1, 1].set_xlabel("Price Range")
        axes[1, 1].set_ylabel("Average Absolute Percentage Difference (%)")
        axes[1, 1].grid(axis="y", linestyle="--", alpha=0.7)

        # Plot 5: Scatter plot of Custom Min vs Professional Min with color gradient
        min_diff = abs(
            results_df["Custom Min Estimate"] - results_df["Estimated Minimum Price"]
        )
        scatter = axes[2, 0].scatter(
            results_df["Estimated Minimum Price"],
            results_df["Custom Min Estimate"],
            c=min_diff,
            cmap="coolwarm",
            alpha=0.7,
        )
        axes[2, 0].plot(
            [
                results_df["Estimated Minimum Price"].min(),
                results_df["Estimated Minimum Price"].max(),
            ],
            [
                results_df["Estimated Minimum Price"].min(),
                results_df["Estimated Minimum Price"].max(),
            ],
            color="green",
            linestyle="--",
            label="Perfect Correlation",
        )
        axes[2, 0].set_title("Custom Min vs Professional Min")
        axes[2, 0].set_xlabel("Professional Minimum Price")
        axes[2, 0].set_ylabel("Custom Minimum Price")
        axes[2, 0].legend()
        axes[2, 0].grid(axis="both", linestyle="--", alpha=0.7)
        fig.colorbar(scatter, ax=axes[2, 0], label="Absolute Difference")

        # Plot 6: Scatter plot of Custom Max vs Professional Max with color gradient
        max_diff = abs(
            results_df["Custom Max Estimate"] - results_df["Estimated Maximum Price"]
        )
        scatter = axes[2, 1].scatter(
            results_df["Estimated Maximum Price"],
            results_df["Custom Max Estimate"],
            c=max_diff,
            cmap="coolwarm",
            alpha=0.7,
        )
        axes[2, 1].plot(
            [
                results_df["Estimated Maximum Price"].min(),
                results_df["Estimated Maximum Price"].max(),
            ],
            [
                results_df["Estimated Maximum Price"].min(),
                results_df["Estimated Maximum Price"].max(),
            ],
            color="green",
            linestyle="--",
            label="Perfect Correlation",
        )
        axes[2, 1].set_title("Custom Max vs Professional Max")
        axes[2, 1].set_xlabel("Professional Maximum Price")
        axes[2, 1].set_ylabel("Custom Maximum Price")
        axes[2, 1].legend()
        axes[2, 1].grid(axis="both", linestyle="--", alpha=0.7)
        fig.colorbar(scatter, ax=axes[2, 1], label="Absolute Difference")

        # Plot 7: Box plot of Custom Min vs Professional Min differences by price range
        sns.boxplot(
            x="Price Range",
            y="Custom Min vs Professional Min Diff",
            data=results_df,
            ax=axes[3, 0],
            color="skyblue",
        )
        axes[3, 0].set_title("Custom Min vs Professional Min Differences")
        axes[3, 0].set_xlabel("Price Range")
        axes[3, 0].set_ylabel("Percentage Difference")
        axes[3, 0].grid(axis="y", linestyle="--", alpha=0.7)

        # Plot 8: Box plot of Custom Max vs Professional Max differences by price range
        sns.boxplot(
            x="Price Range",
            y="Custom Max vs Professional Max Diff",
            data=results_df,
            ax=axes[3, 1],
            color="lightcoral",
        )
        axes[3, 1].set_title("Custom Max vs Professional Max Differences")
        axes[3, 1].set_xlabel("Price Range")
        axes[3, 1].set_ylabel("Percentage Difference")
        axes[3, 1].grid(axis="y", linestyle="--", alpha=0.7)

        # Adjust layout and show the plot
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        plt.subplots_adjust(hspace=0.4)
        if table_name != "Table 3":
            plt.show(block=False)
        else:
            plt.show()
