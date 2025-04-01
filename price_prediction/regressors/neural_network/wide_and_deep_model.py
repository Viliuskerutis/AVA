import torch.nn as nn

from price_prediction.regressors.neural_network.base_model import BaseModel


class WideAndDeepModel(BaseModel):
    def __init__(self, input_size, hidden_units):
        super().__init__(input_size, hidden_units)
        # Wide component
        self.wide = nn.Linear(input_size, 1)
        # Deep component
        self.deep = nn.Sequential(
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
        wide_out = self.wide(x)
        deep_out = self.deep(x)
        return wide_out + deep_out  # Combine wide and deep outputs
