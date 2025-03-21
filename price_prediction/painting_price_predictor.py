from typing import Dict, Any, List, Optional, Union
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split

from data_processing.data_filter_pipeline import process_keep_relevant
from helpers.file_manager import FileManager
from helpers.property_modifier import PropertyModifier
from price_prediction.embedding_model_type import EmbeddingModelType
from price_prediction.regressors.base_regressor import BaseRegressor


class PaintingPricePredictor:
    """
    Handles painting price prediction with preprocessing, feature extraction, and regressor selection.
    """

    NUMERIC_COLUMNS = [
        # "Artist Birth Year",
        # "Artist Death Year",
        # "Creation Year",
        "Width",
        "Height",
        "Years from auction till now",
        "Years from creation till auction",
        # "Years from creation till auction",
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
        regressor: BaseRegressor,
        use_separate_numeric_features: bool,
        encode_per_column: bool,
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
        self.regressor = regressor
        self.use_separate_numeric_features = use_separate_numeric_features
        self.encode_per_column = encode_per_column
        self.use_artfacts = use_artfacts
        self.use_images = use_images
        self.use_count = use_count
        self.use_synthetic = use_synthetic
        self.use_artsy = use_artsy
        self.model = SentenceTransformer(embedding_model_type.value)

        self.artist_info_path = artist_info_path
        self.image_features_path = image_features_path
        self.artist_count_path = artist_count_path
        self.synthetic_paths = synthetic_paths
        self.artsy_path = artsy_path

        self.additional_text_columns = []
        self.additional_numeric_columns = []

    def fill_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensures all columns have non-null values (text columns get 'Unknown', numeric get 0).
        """
        return df.fillna({"": "Unknown"}).fillna(0).reset_index(drop=True)

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

        pca = PCA(n_components=self.use_images)
        reduced_features = pca.fit_transform(features_df)
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
                df[
                    text_columns
                    + ([] if self.use_separate_numeric_features else numeric_columns)
                ]
                .astype(str)
                .agg(" - ".join, axis=1)
            )
            embeddings = self.model.encode(descriptions.tolist()).astype("float32")

        if self.use_separate_numeric_features:
            numeric_data = StandardScaler().fit_transform(df[numeric_columns])
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
                df,
                "Artist name",
                self.artist_info_path,
                "Name",
                ["Birth Year", "Death Year"],
            )
            self.additional_text_columns += art_text
            self.additional_numeric_columns += art_numeric

        if self.use_count:
            df, count_text, count_numeric = PropertyModifier.add_additional_data(
                df, "Artist name", self.artist_count_path, "name", ["path"]
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
                df,
                "Artist name",
                self.synthetic_paths[0],
                "Artist name",
                ["Score Explanations"],
            )
            self.additional_text_columns += synth_text
            self.additional_numeric_columns += synth_numeric

            df, synth_text2, synth_numeric2 = PropertyModifier.add_additional_data(
                df,
                "Auction House",
                self.synthetic_paths[1],
                "Auction House",
                ["Score Explanations"],
            )
            self.additional_text_columns += synth_text2
            self.additional_numeric_columns += synth_numeric2

        if self.use_artsy:
            df, artsy_text, artsy_numeric = PropertyModifier.add_additional_data(
                df, "Artist name", self.artsy_path, "Name", []
            )
            self.additional_text_columns += artsy_text
            self.additional_numeric_columns += artsy_numeric

        # TODO: Move hardcoded values
        if self.use_count:
            # Additional filtering to keep relevant artists only (with no less than `min_artwork_count`)
            df = process_keep_relevant(df, min_artwork_count=None, verbose=False)

        df = self.fill_missing_values(df)
        return df

    def train(self, df: pd.DataFrame) -> None:
        """
        Preprocesses data, extracts features, and trains the regressor.
        """
        self.regressor.clear_fit()
        df = self.preprocess_data(df)
        X = self.generate_combined_embeddings(df)
        y = df["Sold Price"].astype(float)

        # Store the feature columns and types
        self.regressor.feature_columns = df.columns.tolist()
        self.regressor.feature_types = df.dtypes.to_dict()

        self.regressor.train(X, y)

    def predict_single_painting(
        self, row: Union[Dict[str, Any], pd.Series]
    ) -> Optional[float]:
        """
        Preprocesses a single painting's data and predicts its price.
        Accepts either a dictionary or a pandas Series.
        """
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

        return self.regressor.predict(X)[0]

    def filter_prediction_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensures that the DataFrame df contains only the feature columns that were used during training.
        If a column is missing, it is added with a default value based on its type.
        Then, casts each column to its expected dtype.
        Uses schema stored on self.regressor.
        """
        # Verify that the regressor holds the necessary schema.
        if not hasattr(self.regressor, "feature_columns") or not hasattr(
            self.regressor, "feature_types"
        ):
            raise ValueError(
                "Feature columns or types are not stored in the regressor. "
                "Ensure the model is trained or loaded correctly."
            )

        # Add missing columns with default values.
        for col in self.regressor.feature_columns:
            if col not in df.columns:
                expected_dtype = self.regressor.feature_types[col]
                if pd.api.types.is_numeric_dtype(expected_dtype):
                    df[col] = 0
                else:
                    df[col] = "Unknown"

        # Keep only the expected columns and reorder them.
        df = df[self.regressor.feature_columns]

        # Cast each column to its expected type.
        for col in self.regressor.feature_columns:
            df[col] = df[col].astype(self.regressor.feature_types[col])

        return df

    # TODO: Move to experimentation
    def predict_with_test_split(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocesses data, performs train-test split, trains regressor, and evaluates performance.
        """
        self.regressor.clear_fit()
        df = self.preprocess_data(df)

        X, y = self.generate_combined_embeddings(df), df["Sold Price"].astype(float)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.regressor.train(X_train, y_train)
        y_pred = self.regressor.predict(X_test)

        print(f"MAE: {mean_absolute_error(y_test, y_pred)}")
        print(f"MSE: {mean_squared_error(y_test, y_pred)}")
        print(f"MAPE: {mean_absolute_percentage_error(y_test, y_pred)}")
        print(f"R2: {r2_score(y_test, y_pred)}")

        return pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
