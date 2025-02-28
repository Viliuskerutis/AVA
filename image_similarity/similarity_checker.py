import numpy as np
from typing import Dict, Optional
from image_similarity.feature_extractor import FeatureExtractor
from PIL import Image


class SimilarityChecker:
    """Handles finding the most similar image using cosine similarity."""

    def __init__(self, feature_extractor: FeatureExtractor):
        """
        Initializes the SimilarityChecker.

        Args:
            feature_extractor (FeatureExtractor): Feature extractor instance.
        """
        self.feature_extractor = feature_extractor

    def find_similar_above_threshold(
        self, image: Image.Image, features_dict: Dict[str, np.ndarray], threshold: float
    ) -> Optional[str]:
        """
        Finds the most similar image above a given similarity threshold.

        Args:
            image (Image.Image): Loaded input image.
            features_dict (Dict[str, np.ndarray]): Dictionary of stored features.
            threshold (float): Similarity threshold.

        Returns:
            Optional[str]: Photo ID of the most similar image, or None if no match is found.
        """
        new_image_features = self.feature_extractor.extract_features(image)

        # Normalize extracted features before similarity calculation
        new_image_features /= np.linalg.norm(new_image_features, axis=1, keepdims=True)

        best_photo_id = None
        highest_avg_similarity = -1

        for photo_id, stored_features in features_dict.items():
            # Normalize stored features before similarity calculation
            stored_features /= np.linalg.norm(stored_features, axis=1, keepdims=True)

            # Perform cosine similarity check per feature
            similarity_scores = np.einsum(
                "id,id->i", new_image_features, stored_features
            )
            avg_similarity = np.mean(similarity_scores)

            if avg_similarity >= threshold and avg_similarity > highest_avg_similarity:
                highest_avg_similarity = avg_similarity
                best_photo_id = photo_id

        return best_photo_id if highest_avg_similarity >= threshold else None
