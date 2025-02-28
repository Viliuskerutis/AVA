from image_similarity.similarity_model_type import SimilarityModelType

# Experiment Paths
EXPERIMENT_DATA_DIR = (
    "image_similarity/experiments/data"  # Gets the directory of this file
)
EXPERIMENT_MODIFIED_FEATURES_PATH = f"{EXPERIMENT_DATA_DIR}/features-modified.pkl"
EXPERIMENT_MATRICES_PATH = f"{EXPERIMENT_DATA_DIR}/matrices.pkl"

# Similarity Model for Experiments
EXPERIMENT_SIMILARITY_MODEL_NAME = SimilarityModelType.GOOGLENET

# Random Transformations Settings
SCALE_RANGE = (0.9, 1.1)  # Scaling factor range (90% to 110% of original size)
ROTATION_RANGE = (-1.5, 1.5)  # Rotation angle range in degrees
BRIGHTNESS_RANGE = (0.8, 1.2)  # Brightness adjustment range
