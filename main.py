import os
from typing import Dict
from pandas import DataFrame
from tqdm import tqdm
from config import (
    ARTIST_COUNT_CSV_PATH,
    ARTIST_INFORMATION_CSV_PATH,
    ARTIST_SYNTHETIC_CSV_PATH,
    ARTSY_CSV_PATH,
    AUCTION_HOUSE_SYNTHETIC_CSV_PATH,
    IMAGE_FEATURES_PKL_PATH,
    IMAGES_PATH,
    CSVS_PATH,
    SIMILARITY_MODEL_NAME,
    DEVICE,
    OPTIMAL_SIMILARITY_THRESHOLD,
)
from data_processing.data_filter_pipeline import (
    process_for_image_similarity,
    process_keep_relevant,
    process_after_scraping,
)
from image_similarity.model_loader import ModelLoader
from image_similarity.feature_extractor import FeatureExtractor
from image_similarity.similarity_checker import SimilarityChecker
from helpers.file_manager import FileManager
from price_prediction.embedding_model_type import EmbeddingModelType
from price_prediction.painting_price_predictor import PaintingPricePredictor
from price_prediction.regressors.densenet_regressor import DenseNetRegressor
from price_prediction.regressors.histogram_gradient_boosting_regressor import (
    HistogramGradientBoostingRegressor,
)
from price_prediction.regressors.knn_regressor import KNNRegressor
from price_prediction.regressors.lightgbm_regressor import LightGBMRegressor
import pandas as pd
from price_prediction.regressors.mape_meta_regressor import MAPEMetaRegressor
from price_prediction.regressors.neural_midpoint_network_regressor import (
    NeuralMidpointNetworkRegressor,
)
from price_prediction.regressors.neural_network.basic_model import BasicModel
from price_prediction.regressors.neural_network.combined_loss import CombinedLoss
from price_prediction.regressors.neural_network.mape_loss import MAPELoss
from price_prediction.regressors.neural_network.residual_model import ResidualModel
from price_prediction.regressors.neural_network.wide_and_deep_model import (
    WideAndDeepModel,
)
from price_prediction.regressors.neural_network_regressor import NeuralNetworkRegressor
from price_prediction.regressors.random_forest_regressor import (
    RandomForestCustomRegressor,
)
from price_prediction.painting_price_ensemble_predictor import (
    PaintingPriceEnsemblePredictor,
)
from torch import nn
from price_prediction.regressors.decision_tree_custom_regressor import (
    DecisionTreeCustomRegressor,
)
from price_prediction.regressors.weighted_knn_regressor import WeightedKNNRegressor


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
    predictor: PaintingPricePredictor | PaintingPriceEnsemblePredictor,
    data_df: pd.DataFrame,
    painting_data_dict: Dict[str, str],
    use_cross_validation: bool = False,
    use_test_split: bool = True,
    force_retrain: bool = False,
) -> float:
    # For experimentation (skips loading weights, retrains each time, use one or the other)
    # predictor.predict_with_test_split(data_df, test_size=0.3)
    # predictor.cross_validate(data_df, k=5)
    # predictor.train_with_same_data(data_df)

    if use_cross_validation:
        predictor.cross_validate(data_df, k=5)

    model_path = get_predictor_regressor_path(predictor)
    if force_retrain or not predictor.load_model(model_path):
        print(
            "No regressor model found or force retrain enabled. Training and saving new model..."
        )
        if use_cross_validation:
            predictor.regressor.clear_fit()
            predictor.train(data_df)
        else:
            if use_test_split:
                prediction_results = predictor.train_with_test_split(
                    data_df, test_size=0.3
                )
            else:
                prediction_results = predictor.train_with_same_data(data_df)

        predictor.save_model(model_path)

    # predictor.evaluate_custom_estimates(prediction_results)
    # predicted_price = predictor.predict_random_paintings(data_df, count=100)

    predicted_price = predictor.predict_single_painting(painting_data_dict)
    print(f'Random Painting Sold Price: {painting_data_dict["Sold Price"]}')
    print(f"Predicted Sold Price: {predicted_price:.2f}")

    return predicted_price


def get_predictor_regressor_path(predictor):
    if isinstance(predictor, PaintingPricePredictor):
        model_path = predictor.regressor.path
    elif isinstance(predictor, PaintingPriceEnsemblePredictor):
        directory = os.path.dirname(predictor.regressors[0].path)
        regressor_filenames = [
            os.path.splitext(os.path.basename(regressor.path))[0]
            for regressor in predictor.regressors
        ]
        combined_filename = f"{'_and_'.join(regressor_filenames)}.pkl"
        model_path = os.path.join(directory, combined_filename)
    else:
        raise ValueError("Unsupported predictor type")
    return model_path


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
        data_for_prediction_df = process_after_scraping(data_df)
        # Additional filtering to keep relevant artists only (set possible price range for predictions for higher accuracy)
        data_for_prediction_df = process_keep_relevant(
            data_for_prediction_df,
            min_price=None,
            max_price=None,
            max_missing_percent=None,
            filter_by_column=None,
        )

        # Initialization of regressor and predictor can be done once when server is run
        # while `predict_price()` can be called per user request for performance

        # regressor = KNNRegressor(n_neighbors=5)
        # regressor = RandomForestCustomRegressor(n_estimators=20)
        regressor2 = LightGBMRegressor(n_estimators=500)
        regressor1 = NeuralNetworkRegressor(
            model_class=WideAndDeepModel,
            loss_function=CombinedLoss(
                MAPELoss(), nn.L1Loss(), alpha=0.9
            ),  # nn.MSELoss(), nn.L1Loss(),  # MAPELoss(),  #CombinedLoss()
            hidden_units=1024,
            learning_rate=0.001,
            epochs=1000,
            batch_size=16,
            patience=15,
            lr_patience=5,
            lr_factor=0.25,
        )
        # regressor1 = NeuralMidpointNetworkRegressor(
        #     model_class=WideAndDeepModel,
        #     loss_function=MidpointDeviationLoss(),
        #     hidden_units=1024,
        #     learning_rate=0.001,
        #     epochs=1000,
        #     batch_size=16,
        #     patience=15,
        #     lr_patience=5,
        #     lr_factor=0.25,
        # )
        # regressor = HistogramGradientBoostingRegressor()
        # regressor = DenseNetRegressor(
        #     loss_function=nn.L1Loss(),
        #     learning_rate=0.001,
        #     epochs=1000,
        #     batch_size=32,
        #     patience=8,
        #     lr_patience=3,
        #     lr_factor=0.25,
        # )
        # regressor1 = WeightedKNNRegressor(
        #     n_neighbors=7, weights="distance", algorithm="auto", p=2
        # )
        # regressor = DecisionTreeCustomRegressor(max_depth=20)

        predictor = PaintingPriceEnsemblePredictor(
            regressors=[regressor1, regressor2],
            meta_regressor=MAPEMetaRegressor(),
            max_missing_percent=0.15,  # Set to 1.0 to keep missing data filled with "Unknown" and -1
            use_separate_numeric_features=True,
            encode_per_column=False,
            hot_encode_columns=["Surface", "Materials"],
            use_artfacts=False,
            use_images=50,
            use_count=True,
            use_synthetic=True,
            use_artsy=False,
            embedding_model_type=EmbeddingModelType.ALL_MPNET_BASE,
            artist_info_path=ARTIST_INFORMATION_CSV_PATH,
            image_features_path=IMAGE_FEATURES_PKL_PATH,
            artist_count_path=ARTIST_COUNT_CSV_PATH,
            synthetic_paths=[
                ARTIST_SYNTHETIC_CSV_PATH,
                AUCTION_HOUSE_SYNTHETIC_CSV_PATH,
            ],
            artsy_path=ARTSY_CSV_PATH,
        )
        # predictor = PaintingPricePredictor(
        #     regressor=regressor,
        #     max_missing_percent=0.15,  # Set to 1.0 to keep missing data filled with "Unknown" and -1
        #     use_separate_numeric_features=True,
        #     encode_per_column=False,
        #     hot_encode_columns=["Surface", "Materials"],
        #     use_artfacts=False,
        #     use_images=50,
        #     use_count=True,
        #     use_synthetic=True,
        #     use_artsy=False,
        #     embedding_model_type=EmbeddingModelType.ALL_MPNET_BASE,
        #     artist_info_path=ARTIST_INFORMATION_CSV_PATH,
        #     image_features_path=IMAGE_FEATURES_PKL_PATH,
        #     artist_count_path=ARTIST_COUNT_CSV_PATH,
        #     synthetic_paths=[
        #         ARTIST_SYNTHETIC_CSV_PATH,
        #         AUCTION_HOUSE_SYNTHETIC_CSV_PATH,
        #     ],
        #     artsy_path=ARTSY_CSV_PATH,
        # )

        # for i in range(0, 50):
        #     random_painting_data = (
        #         data_for_prediction_df.sample(n=1, random_state=i).iloc[0].to_dict()
        #     )
        #     predicted_price = predict_price(
        #         predictor,
        #         data_for_prediction_df,
        #         random_painting_data,
        #         force_retrain=True,
        #     )

        # This should be the user painting information from UI (temporarily picking random value)
        # preprocessed_df = predictor.preprocess_data(data_for_prediction_df, True)
        random_painting_data = (
            data_for_prediction_df.sample(n=1, random_state=21).iloc[0].to_dict()
        )
        # random_painting_data = (
        #     data_for_prediction_df[
        #         (data_for_prediction_df["Artist name"] == "Tom BYRNE")
        #         & (data_for_prediction_df["Painting name"] == "Eminem")
        #     ]
        #     .iloc[0]
        #     .to_dict()
        # )

        predicted_price = predict_price(
            predictor,
            data_for_prediction_df,
            random_painting_data,
            use_test_split=True,
            force_retrain=True,
        )
