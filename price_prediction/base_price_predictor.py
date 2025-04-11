from abc import ABC, abstractmethod
from typing import List

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler

from data_processing.data_filter_pipeline import (
    ensure_data_filled_and_correct,
    process_keep_relevant,
)
from helpers.file_manager import FileManager
from helpers.property_modifier import PropertyModifier
from price_prediction.embedding_model_type import EmbeddingModelType


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

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies the full preprocessing pipeline: filling missing data, removing outliers,
        adding images, and enriching with additional data.
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

        df = process_keep_relevant(df, max_missing_percent=self.max_missing_percent)
        return df
