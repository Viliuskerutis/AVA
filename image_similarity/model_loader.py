import torch
import torch.nn as nn
from torchvision.models import resnet50, googlenet, ResNet50_Weights, GoogLeNet_Weights
from image_similarity.similarity_model_type import SimilarityModelType


class ModelLoader:
    """Loads pre-trained models and removes the final classification layer."""

    @staticmethod
    def load_model(model_type: SimilarityModelType, device: torch.device) -> nn.Module:
        """
        Loads the specified model and removes its final fully connected layer.

        Args:
            model_type (ModelType): The type of model to load (ResNet50 or GoogLeNet).
            device (torch.device): The device to load the model onto.

        Returns:
            nn.Module: The modified model without the classification layer.
        """
        if model_type == SimilarityModelType.RESNET50:
            model = resnet50(weights=ResNet50_Weights.DEFAULT)
        elif model_type == SimilarityModelType.GOOGLENET:
            model = googlenet(weights=GoogLeNet_Weights.DEFAULT)
        else:
            raise ValueError(f"Unsupported model: {model_type}")

        model = nn.Sequential(*list(model.children())[:-1])  # Remove FC layer
        return model.to(device).eval()
