from abc import ABC, abstractmethod
import torch.nn as nn


class BaseModel(nn.Module, ABC):
    @abstractmethod
    def __init__(self, input_size, hidden_units):
        """
        Initialize the model with the required parameters.
        """
        super().__init__()

    @abstractmethod
    def forward(self, x):
        """
        Define the forward pass of the model.
        """
        pass
