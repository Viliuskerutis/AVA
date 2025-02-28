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
ARTIST_INFORMATION_CSV_PATH = f"{ADDITIONAL_DATA_PATH}/artist_data.csv"
COMBINED_INFORMATION_CSV_PATH = f"{ADDITIONAL_DATA_PATH}/combined_filtered_data.csv"
