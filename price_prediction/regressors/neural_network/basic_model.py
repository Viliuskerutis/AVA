import torch.nn as nn

from price_prediction.regressors.neural_network.base_model import BaseModel


class BasicModel(BaseModel):
    def __init__(self, input_size, hidden_units):
        super().__init__(input_size, hidden_units)
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_units),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_units),
            nn.Dropout(0.3),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_units),
            nn.Dropout(0.3),
            nn.Linear(hidden_units, 1),
        )

    def forward(self, x):
        return self.model(x)
