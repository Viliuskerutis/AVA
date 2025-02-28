import os
import pickle
from tqdm import tqdm
import numpy as np
from config import IMAGE_FEATURES_PKL_PATH, IMAGES_PATH, CSVS_PATH, DEVICE
from image_similarity.experiments.experiment_analysis import ExperimentAnalysis
from image_similarity.experiments.experiment_config import (
    EXPERIMENT_MODIFIED_FEATURES_PATH,
    EXPERIMENT_MATRICES_PATH,
    EXPERIMENT_SIMILARITY_MODEL_NAME,
)
from helpers.file_manager import FileManager
from image_similarity.model_loader import ModelLoader
from image_similarity.feature_extractor import FeatureExtractor
from image_similarity.experiments.transformations import ImageTransformations
from image_similarity.similarity_model_type import SimilarityModelType


class ExperimentRunner:
    """Handles experimentation by extracting features, modifying images, and computing similarity matrices."""

    def __init__(self, model: SimilarityModelType):
        """Initialize model and feature extractor."""
        self.model = ModelLoader.load_model(model, DEVICE)
        self.feature_extractor = FeatureExtractor(self.model)

    def extract_and_save_features(self, save_path: str, apply_transformations=False):
        """
        Extracts and saves image features **only if the file does not exist**.

        Args:
            save_path (str): Path to save extracted features.
            apply_transformations (bool): Whether to apply random modifications.
        """
        if os.path.exists(save_path):
            print(f"Skipping extraction. Features already exist at {save_path}")
            return  # Skip extraction

        print(
            f"Extracting {'modified' if apply_transformations else 'original'} features..."
        )

        image_paths = FileManager.get_full_image_paths(IMAGES_PATH)
        data_df = FileManager.read_all_csvs(CSVS_PATH)

        features = {}

        for _, row in tqdm(
            data_df.iterrows(), total=len(data_df), desc="Extracting Features"
        ):
            photo_id = row["Photo id"]
            if photo_id in image_paths:
                image = FileManager.load_image(image_paths[photo_id])

                if apply_transformations:
                    image = ImageTransformations.apply_random_transformations(image)

                features[photo_id] = self.feature_extractor.extract_features(image)

        FileManager.save_to_pkl(features, save_path)
        print(f"Features saved to {save_path}")

    def calculate_and_save_similarity_matrices(self):
        """
        Computes similarity matrices if needed, otherwise loads them.
        Always returns min, max, avg similarity matrices.

        Returns:
            tuple: (avg_similarity_matrix, min_similarity_matrix, max_similarity_matrix, image_names)
        """
        if os.path.exists(EXPERIMENT_MATRICES_PATH):
            print(
                f"Loading existing similarity matrices from {EXPERIMENT_MATRICES_PATH}"
            )
            with open(EXPERIMENT_MATRICES_PATH, "rb") as f:
                matrices = pickle.load(f)
            return (
                matrices["avg_similarity_matrix"],
                matrices["min_similarity_matrix"],
                matrices["max_similarity_matrix"],
                matrices["image_names"],
            )

        print("Computing similarity matrices between non-modified and modified images.")

        features_original = FileManager.load_from_pkl(IMAGE_FEATURES_PKL_PATH)
        features_modified = FileManager.load_from_pkl(EXPERIMENT_MODIFIED_FEATURES_PATH)

        if not features_original or not features_modified:
            raise ValueError("Feature files are missing. Run feature extraction first.")

        feature_tensor_original = np.array(
            [np.stack(features) for features in features_original.values()]
        )
        feature_tensor_modified = np.array(
            [np.stack(features) for features in features_modified.values()]
        )

        # Normalize each feature vector
        normalized_features_original = feature_tensor_original / np.linalg.norm(
            feature_tensor_original, axis=2, keepdims=True
        )
        normalized_features_modified = feature_tensor_modified / np.linalg.norm(
            feature_tensor_modified, axis=2, keepdims=True
        )

        # Compute pairwise cosine similarity
        similarity_tensor = np.einsum(
            "icd,jcd->ijc", normalized_features_original, normalized_features_modified
        )

        # Clip similarity values to the range [0, 1] for numerical stability
        similarity_tensor = np.clip(similarity_tensor, 0, 1)

        # Aggregate similarity across crops
        avg_similarity_matrix = similarity_tensor.mean(axis=2)
        min_similarity_matrix = similarity_tensor.min(axis=2)
        max_similarity_matrix = similarity_tensor.max(axis=2)

        image_names = list(features_original.keys())

        # Save matrices
        self.save_similarity_matrices(
            EXPERIMENT_MATRICES_PATH,
            avg_similarity_matrix,
            min_similarity_matrix,
            max_similarity_matrix,
            image_names,
        )

        return (
            avg_similarity_matrix,
            min_similarity_matrix,
            max_similarity_matrix,
            image_names,
        )

    def save_similarity_matrices(
        self, file_path, avg_matrix, min_matrix, max_matrix, image_names
    ):
        """
        Saves similarity matrices.

        Args:
            file_path (str): Path to save the matrices.
            avg_matrix (np.ndarray): Average similarity matrix.
            min_matrix (np.ndarray): Minimum similarity matrix.
            max_matrix (np.ndarray): Maximum similarity matrix.
            image_names (list): List of image names.
        """
        with open(file_path, "wb") as f:
            pickle.dump(
                {
                    "avg_similarity_matrix": avg_matrix,
                    "min_similarity_matrix": min_matrix,
                    "max_similarity_matrix": max_matrix,
                    "image_names": image_names,
                },
                f,
            )
        print(f"Similarity matrices saved to {file_path}")


if __name__ == "__main__":
    runner = ExperimentRunner(EXPERIMENT_SIMILARITY_MODEL_NAME)

    runner.extract_and_save_features(
        IMAGE_FEATURES_PKL_PATH, apply_transformations=False
    )
    runner.extract_and_save_features(
        EXPERIMENT_MODIFIED_FEATURES_PATH, apply_transformations=True
    )

    avg_matrix, min_matrix, max_matrix, image_names = (
        runner.calculate_and_save_similarity_matrices()
    )

    analysis_avg = ExperimentAnalysis(avg_matrix, image_names)
    analysis_avg.run_analysis("Average Similarity")

    # print("Experiment completed.")
