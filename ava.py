from flask import Flask, jsonify, request
from flask_cors import CORS
from helpers.file_manager import FileManager
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
    OPTIMAL_SIMILARITY_THRESHOLD
)
from data_processing.data_filter_pipeline import (
    process_for_image_similarity,
    process_keep_relevant,
    process_after_scraping
)
from main import (
    try_find_most_similar,
    predict_price
)
import os
import json

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

# Create a Flask application
app = Flask(__name__)

# Enable CORS for specific origins
CORS(app, origins=["http://localhost:5173", "https://voi.lt", "https://artvaluation.app"])

# Folder to save uploaded images
UPLOAD_FOLDER = 'data/images/runtime'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Create folder if not exists
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Define a route for the root endpoint
@app.route('/', methods=['GET'])
def home():
    return "Hello, this is your server responding!"

# Define a route to receive an image file
@app.route('/evaluate', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    # Save the file locally
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    # from main.py
    # First part - if identical image exists, return data (price) of painting
    image_path_dict = FileManager.get_full_image_paths(IMAGES_PATH)
    data_df = FileManager.read_all_csvs(CSVS_PATH)

    # Keep values that have price, remove non-existant images, remove duplicate photo ids, remove low quality
    data_for_similarity_df = process_for_image_similarity(data_df, image_path_dict)

    input_image_path = file_path
    most_similar = try_find_most_similar(
        image_path_dict, data_for_similarity_df, input_image_path
    )

    if most_similar is None:
        try:
            # In current implementation it is expected that image from `input_image_path` exists in data folder
            # User input data should be passed trough processing pipeline before use
            data_for_prediction_df = process_after_scraping(data_df)

            data_for_prediction_df = process_keep_relevant(
            data_for_prediction_df,
            min_price=None,
            max_price=None,
            max_missing_percent=None,
            filter_by_column=None,
        )
            
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

            # This should be the user painting information from UI (temporarily picking random value)
            random_painting_data = data_for_prediction_df.sample(n=1).iloc[0].to_dict()

            predicted_price = int(round(predict_price(
            predictor,
            data_for_prediction_df,
            random_painting_data,
            use_test_split=True,
            force_retrain=False,
        )))
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    if os.path.exists(file_path):
        os.remove(file_path)
    
    return jsonify(
	{"data": None if most_similar is None else json.loads(most_similar.to_json(orient='records', force_ascii=False)),
    "price": most_similar["Sold Price"].values[0] if most_similar is not None else f"€{predicted_price}",
    "message": "Exact image found inside the storage" if most_similar is not None else "Image price evaluated successfully"}), 200

# Start the server
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
