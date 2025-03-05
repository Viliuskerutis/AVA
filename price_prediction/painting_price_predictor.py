from typing import Dict, Any, List, Optional
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

from helpers.file_manager import FileManager
from price_prediction.embedding_model_type import EmbeddingModelType
from price_prediction.regressors.base_regressor import BaseRegressor


class PaintingPricePredictor:
    """
    Handles painting price prediction with preprocessing, feature extraction, and regressor selection.
    """

    NUMERIC_COLUMNS = [
        "Width",
        "Height",
        "Years from auction till now",
        "Years from creation till auction",
        "Artist Lifetime",
        "Average Sold Price",
        "Min Sold Price",
        "Max Sold Price",
    ]

    ARTFACTS_NUMERIC_COLUMNS = [
        "Birth Year",
        "Death Year",
        "Verified Exhibitions",
        "Solo Exhibitions",
        "Group Exhibitions",
        "Biennials",
        "Art Fairs",
    ]

    TEXT_COLUMNS = [
        "Artist name",
        "Auction House",
        "Auction Country",
        "Materials",
        "Surface",
        "Signed",
    ]

    ARTFACTS_TEXT_COLUMNS = [
        "Ranking",
        "Birth Country",
        "Gender",
        "Nationality",
        "Movement",
    ]

    def __init__(
        self,
        regressor: BaseRegressor,
        use_separate_numeric_features: bool,
        encode_per_column: bool,
        use_artfacts: bool,
        use_images: int,
        embedding_model_type: EmbeddingModelType,
        artist_info_path: str,
        image_features_path: str,
    ):
        self.regressor = regressor
        self.use_separate_numeric_features = use_separate_numeric_features
        self.encode_per_column = encode_per_column
        self.use_artfacts = use_artfacts
        self.use_images = use_images
        self.model = SentenceTransformer(embedding_model_type.value)

        self.artist_info_path = artist_info_path
        self.image_features_path = image_features_path

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

    def remove_price_outliers(
        self, df: pd.DataFrame, column="Sold Price", method="iqr", factor=1.5
    ) -> pd.DataFrame:
        """
        Removes outliers from 'Sold Price' column using either IQR or Z-score method.
        """
        if method == "iqr":
            Q1, Q3 = df[column].quantile([0.25, 0.75])
            IQR = Q3 - Q1
            return df[
                (df[column] >= Q1 - factor * IQR) & (df[column] <= Q3 + factor * IQR)
            ]
        elif method == "zscore":
            z_scores = (df[column] - df[column].mean()) / df[column].std()
            return df[abs(z_scores) <= factor]
        else:
            raise ValueError("Unsupported outlier detection method")

    def add_artist_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Merges external artist data into the main dataframe (using artist name as the key).
        """
        artist_data = pd.read_csv(self.artist_info_path, sep=";", encoding="utf-8")
        artist_data["Name"] = artist_data["Name"].str.lower()
        df["Artist name"] = df["Artist name"].str.lower()

        artist_data = self.convert_columns_to_numeric(
            artist_data, self.ARTFACTS_NUMERIC_COLUMNS
        )
        df = self.convert_columns_to_numeric(df, self.NUMERIC_COLUMNS)

        df = df.merge(
            artist_data, left_on="Artist name", right_on="Name", how="left"
        ).drop(columns=["Name"])
        return self.fill_missing_values(df)

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
        text_columns = self.TEXT_COLUMNS + (
            self.ARTFACTS_TEXT_COLUMNS if self.use_artfacts else []
        )
        numeric_columns = self.NUMERIC_COLUMNS + (
            self.ARTFACTS_NUMERIC_COLUMNS if self.use_artfacts else []
        )

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
        Applies the full preprocessing pipeline: filling missing data, removing outliers, adding images, and enriching with artist data.
        """
        df = self.fill_missing_values(df)
        df = self.remove_price_outliers(df)

        if self.use_images:
            df = self.add_image_features(df)

        if self.use_artfacts:
            df = self.add_artist_data(df)

        return df

    def train(self, df: pd.DataFrame) -> None:
        """
        Preprocesses data, extracts features, and trains the regressor.
        """
        self.regressor.clear_fit()
        df = self.preprocess_data(df)

        X, y = self.generate_combined_embeddings(df), df["Sold Price"].astype(float)
        self.regressor.train(X, y)

    def predict_single_painting(self, row: Dict[str, Any]) -> Optional[float]:
        """
        Preprocesses a single painting's data and predicts its price.
        """
        df_single = pd.DataFrame([row])
        df_single = self.preprocess_data(df_single)
        X = self.generate_combined_embeddings(df_single)
        return self.regressor.predict(X)[0]

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
