from typing import Dict
from pandas import DataFrame
from tqdm import tqdm
from config import (
    ARTIST_INFORMATION_CSV_PATH,
    COMBINED_INFORMATION_CSV_PATH,
    IMAGE_FEATURES_PKL_PATH,
    IMAGES_PATH,
    CSVS_PATH,
    SIMILARITY_MODEL_NAME,
    DEVICE,
    OPTIMAL_SIMILARITY_THRESHOLD,
)

from image_similarity.model_loader import ModelLoader
from image_similarity.feature_extractor import FeatureExtractor
from image_similarity.similarity_checker import SimilarityChecker
from helpers.file_manager import FileManager
from price_prediction.embedding_model_type import EmbeddingModelType
from price_prediction.lightgbm_regressor import LightGBMRegressor
from price_prediction.painting_price_predictor import PaintingPricePredictor


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
    painting_data_dict: Dict[str, str],
) -> float:
    # filtered_data_df = process_initial_cleanup(data_df, image_path_dict)
    # === SOME LOGIC TO CONVERT COLUMNS === (WILL BE ADDED LATER, LOADING PRESET FILE INSTEAD)

    filtered_final_df = FileManager.read_single_csv(COMBINED_INFORMATION_CSV_PATH)

    # knn_regressor = KNNRegressor(n_neighbors=5)
    # random_forest_regressor = RandomForestCustomRegressor(n_estimators=10)
    lightgbm_regressor = LightGBMRegressor(n_estimators=500)

    predictor = PaintingPricePredictor(
        regressor=lightgbm_regressor,
        use_separate_numeric_features=True,
        encode_per_column=True,
        use_artfacts=True,
        use_images=0,
        embedding_model_type=EmbeddingModelType.ALL_MINILM,
        artist_info_path=ARTIST_INFORMATION_CSV_PATH,
        image_features_path=IMAGE_FEATURES_PKL_PATH,
    )

    # For experimentation
    # print(predictor.predict_with_test_split(filtered_final_df))

    predictor.train(filtered_final_df)

    predicted_price = predictor.predict_single_painting(painting_data_dict)

    return predicted_price


if __name__ == "__main__":
    # First part - if identical image exists, return data (price) of painting
    image_path_dict = FileManager.get_full_image_paths(IMAGES_PATH)
    data_df = FileManager.read_all_csvs(CSVS_PATH)
    # This should be the path of user input image from UI (temporarily picking random image from dataset)
    input_image_path = FileManager.get_random_image_path(IMAGES_PATH)
    most_similar = try_find_most_similar(image_path_dict, data_df, input_image_path)
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
    if most_similar is None or price == "Not Sold" or force_prediction:
        # This should be the data of painting from UI (temporarily picking random entry from dataset)
        # In current implementation it is expected that image from `input_image_path` exists in data folder
        # It will need to be additionally filtered/modified to allow feature engineering
        filtered_final_df = FileManager.read_single_csv(COMBINED_INFORMATION_CSV_PATH)
        random_painting_data = filtered_final_df.sample(n=1).iloc[0].to_dict()

        predicted_price = predict_price(random_painting_data)

        print(f'Random Painting Sold Price: {random_painting_data["Sold Price"]}')
        print(f"Predicted Sold Price: {predicted_price:.2f}")
