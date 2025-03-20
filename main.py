from typing import Dict
from pandas import DataFrame
from tqdm import tqdm
from config import (
    ARTIST_COUNT_CSV_PATH,
    ARTIST_INFORMATION_CSV_PATH,
    ARTIST_SYNTHETIC_CSV_PATH,
    AUCTION_HOUSE_SYNTHETIC_CSV_PATH,
    IMAGE_FEATURES_PKL_PATH,
    IMAGES_PATH,
    CSVS_PATH,
    REGRESSOR_LIGHTGBM_PKL_PATH,
    SIMILARITY_MODEL_NAME,
    DEVICE,
    OPTIMAL_SIMILARITY_THRESHOLD,
)

from data_processing.data_filter_pipeline import (
    process_for_image_similarity,
    process_for_predictions,
    process_keep_relevant,
)
from image_similarity.model_loader import ModelLoader
from image_similarity.feature_extractor import FeatureExtractor
from image_similarity.similarity_checker import SimilarityChecker
from helpers.file_manager import FileManager
from price_prediction.embedding_model_type import EmbeddingModelType
from price_prediction.painting_price_predictor import PaintingPricePredictor
from price_prediction.regressors.knn_regressor import KNNRegressor
from price_prediction.regressors.lightgbm_regressor import LightGBMRegressor
import pandas as pd
from price_prediction.regressors.random_forest_regressor import (
    RandomForestCustomRegressor,
)


def try_find_most_similar(
    image_path_dict: Dict[str, str], data_df: DataFrame, input_image_path: str
) -> DataFrame | None:
    input_image = FileManager.load_image(input_image_path)

    model = ModelLoader.load_model(SIMILARITY_MODEL_NAME, DEVICE)
    feature_extractor = FeatureExtractor(model)

    # Check if feature pickle file exists, else extract features
    features = FileManager.load_from_pkl(IMAGE_FEATURES_PKL_PATH)
    if not features:
        features = {}
        for _, row in tqdm(
            data_df.iterrows(), total=len(data_df), desc="Extracting features"
        ):
            photo_id = row["Photo id"]
            if photo_id in image_path_dict:
                image = FileManager.load_image(image_path_dict[photo_id])
                features[photo_id] = feature_extractor.extract_features(image)

        FileManager.save_to_pkl(features, IMAGE_FEATURES_PKL_PATH)

    similarity_checker = SimilarityChecker(feature_extractor)
    most_similar_photo_id = similarity_checker.find_similar_above_threshold(
        input_image, features, OPTIMAL_SIMILARITY_THRESHOLD
    )

    identical_data = (
        data_df.loc[data_df["Photo id"] == most_similar_photo_id]
        if most_similar_photo_id
        else None
    )

    return identical_data


def predict_price(
    predictor: PaintingPricePredictor,
    data_df: pd.DataFrame,
    painting_data_dict: Dict[str, str],
    force_retrain: bool = False,
) -> float:
    # For experimentation (skips loading weights, retrains each time)
    # print(predictor.predict_with_test_split(data_df))

    if force_retrain or not predictor.regressor.load_model(REGRESSOR_LIGHTGBM_PKL_PATH):
        print(
            "No regressor model found or force retrain enabled. Training and saving new model..."
        )
        predictor.train(data_df)
        predictor.regressor.save_model(REGRESSOR_LIGHTGBM_PKL_PATH)

    predicted_price = predictor.predict_single_painting(painting_data_dict)

    return predicted_price


if __name__ == "__main__":
    # First part - if identical image exists, return data (price) of painting
    image_path_dict = FileManager.get_full_image_paths(IMAGES_PATH)
    data_df = FileManager.read_all_csvs(CSVS_PATH)

    # Keep values that have price, remove non-existant images, remove duplicate photo ids, remove low quality
    data_for_similarity_df = process_for_image_similarity(data_df, image_path_dict)

    # This should be the path of user input image from UI (temporarily picking random image from dataset)
    input_image_path = FileManager.get_random_image_path(IMAGES_PATH)
    most_similar = try_find_most_similar(
        image_path_dict, data_for_similarity_df, input_image_path
    )
    if most_similar is not None:
        title = most_similar["Painting name"].values[0]
        artist = most_similar["Artist name"].values[0]
        price = most_similar["Sold Price"].values[0]

        print(
            f"Similar painting titled '{title}' by {artist} was sold at price of {price}"
        )
    else:
        print("No match found.")

    force_prediction = True  # Temporary variable to force price prediction even if identical painting is found
    # Second part - if identical image doesn't exist, try predict price using regressors
    if most_similar is None or force_prediction:
        # In current implementation it is expected that image from `input_image_path` exists in data folder
        # User input data should be passed trough processing pipeline before use
        data_for_prediction_df = process_for_predictions(data_df)
        # Additional filtering to keep relevant artists only (set possible price range for predictions for higher accuracy)
        data_for_prediction_df = process_keep_relevant(
            data_for_prediction_df, min_price=None, max_price=None
        )

        # Initialization of regressor and predictor can be done once when server is run
        # while `predict_price()` can be called per user request for performance

        # regressor = KNNRegressor(n_neighbors=5)
        # regressor = RandomForestCustomRegressor(n_estimators=10)
        regressor = LightGBMRegressor(n_estimators=1000)
        predictor = PaintingPricePredictor(
            regressor=regressor,
            use_separate_numeric_features=True,
            encode_per_column=True,
            use_artfacts=False,
            use_images=0,
            use_count=True,
            use_synthetic=False,
            embedding_model_type=EmbeddingModelType.ALL_MINILM,
            artist_info_path=ARTIST_INFORMATION_CSV_PATH,
            image_features_path=IMAGE_FEATURES_PKL_PATH,
            artist_count_path=ARTIST_COUNT_CSV_PATH,
            synthetic_paths=[
                ARTIST_SYNTHETIC_CSV_PATH,
                AUCTION_HOUSE_SYNTHETIC_CSV_PATH,
            ],
        )

        # This should be the user painting information from UI (temporarily picking random value)
        random_painting_data = data_for_prediction_df.sample(n=1).iloc[0].to_dict()

        predicted_price = predict_price(
            predictor,
            data_for_prediction_df,
            random_painting_data,
            force_retrain=False,
        )

        print(f'Random Painting Sold Price: {random_painting_data["Sold Price"]}')
        print(f"Predicted Sold Price: {predicted_price:.2f}")
