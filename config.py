import torch

from image_similarity.similarity_model_type import SimilarityModelType

# Image similarity configurations
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SIMILARITY_MODEL_NAME = SimilarityModelType.GOOGLENET
SIMILARITY_MODEL_INPUT_SIZE = (224, 224)
CROP_FRACTIONS = [
    (0.15, 0.15, 0.35, 0.3),
    (0.45, 0.65, 0.3, 0.2),
    (0.7, 0.4, 0.15, 0.2),
]
OPTIMAL_SIMILARITY_THRESHOLD = 0.85  # Found via experimentation

# Data Paths
DATA_PATH = "data"
IMAGES_PATH = f"{DATA_PATH}/images"
CSVS_PATH = f"{DATA_PATH}/csvs"
IMAGE_FEATURES_PKL_PATH = f"{DATA_PATH}/image-features.pkl"

ADDITIONAL_DATA_PATH = f"{DATA_PATH}/additional_data"
ARTIST_INFORMATION_CSV_PATH = f"{ADDITIONAL_DATA_PATH}/artfacts_artists.csv"
COMBINED_INFORMATION_CSV_PATH = f"{ADDITIONAL_DATA_PATH}/combined_filtered_data.csv"
ARTIST_COUNT_CSV_PATH = f"{ADDITIONAL_DATA_PATH}/artist_counts.csv"
ARTIST_SYNTHETIC_CSV_PATH = f"{ADDITIONAL_DATA_PATH}/artists_gpto4mini.csv"
AUCTION_HOUSE_SYNTHETIC_CSV_PATH = (
    f"{ADDITIONAL_DATA_PATH}/auction_houses_gpt4omini.csv"
)
ARTSY_CSV_PATH = f"{ADDITIONAL_DATA_PATH}/artsy_artists.csv"
PROCESSED_DATA_PATH = f"{ADDITIONAL_DATA_PATH}/processed_data_default.csv"
PROCESSED_COMBINED_DATA_PATH = f"{ADDITIONAL_DATA_PATH}/processed_combined_data.csv"

# Fitted model weight paths
REGRESSOR_HISTOGRAM_GRADIENT_BOOSTING_PKL_PATH = (
    f"{DATA_PATH}/histogram_gradient_boosting.pkl"
)
REGRESSOR_KNN_PKL_PATH = f"{DATA_PATH}/knn.pkl"
REGRESSOR_LIGHTGBM_PKL_PATH = f"{DATA_PATH}/lightgbm.pkl"
REGRESSOR_NEURAL_NETWORK_PKL_PATH = f"{DATA_PATH}/neural_network.pkl"
REGRESSOR_RANDOM_FOREST_PKL_PATH = f"{DATA_PATH}/random_forest.pkl"
REGRESSOR_NEURAL_MIDPOINT_NETWORK_PKL_PATH = f"{DATA_PATH}/neural_midpoint_network.pkl"
REGRESSOR_WEIGHTED_KNN_PKL_PATH = f"{DATA_PATH}/weighted_knn.pkl"
REGRESSOR_DECISION_TREE_PKL_PATH = f"{DATA_PATH}/decision_tree.pkl"
