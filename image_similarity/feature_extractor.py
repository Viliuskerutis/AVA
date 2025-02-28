import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from config import SIMILARITY_MODEL_INPUT_SIZE, CROP_FRACTIONS, DEVICE


class FeatureExtractor:
    """Handles feature extraction from images using a pre-trained model."""

    def __init__(self, model: torch.nn.Module):
        """
        Initializes the FeatureExtractor with the given model.

        Args:
            model (torch.nn.Module): The pre-trained model for feature extraction.
        """
        self.model = model
        self.preprocess = transforms.Compose(
            [
                transforms.Resize(SIMILARITY_MODEL_INPUT_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def extract_features(self, image: Image.Image) -> np.ndarray:
        """
        Extracts feature vectors from different cropped regions of an image.

        Args:
            image (Image.Image): The input image.

        Returns:
            np.ndarray: Extracted feature vectors of shape (num_crops, feature_dim).
        """
        feature_list = []
        for crop_fraction in CROP_FRACTIONS:
            image_tensor = self._crop_and_preprocess(image, crop_fraction)
            with torch.no_grad():
                features = self.model(image_tensor).squeeze().cpu().numpy()
                feature_list.append(features)
        return np.array(feature_list)

    def _crop_and_preprocess(self, image: Image.Image, crop_fraction) -> torch.Tensor:
        """
        Crops and preprocesses an image for feature extraction.

        Args:
            image (Image.Image): The input image.
            crop_fraction (tuple): The cropping fractions.

        Returns:
            torch.Tensor: The preprocessed image tensor.
        """
        width, height = image.size
        crop_x = int(width * crop_fraction[0])
        crop_y = int(height * crop_fraction[1])
        crop_width = int(width * crop_fraction[2])
        crop_height = int(height * crop_fraction[3])

        cropped_image = image.crop(
            (crop_x, crop_y, crop_x + crop_width, crop_y + crop_height)
        )
        return self.preprocess(cropped_image).unsqueeze(0).to(DEVICE)
