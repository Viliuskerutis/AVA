import torch.nn as nn

from price_prediction.regressors.neural_network.base_model import BaseModel


class ResidualBlock(nn.Module):
    def __init__(self, input_size, hidden_units):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_units)
        self.bn1 = nn.BatchNorm1d(hidden_units)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_units, hidden_units)
        self.bn2 = nn.BatchNorm1d(hidden_units)

    def forward(self, x):
        residual = x
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out += residual  # Add skip connection
        out = self.relu(out)
        return out


class ResidualModel(BaseModel):
    def __init__(self, input_size, hidden_units):
        super().__init__(input_size, hidden_units)
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_units),
            nn.ReLU(),
            ResidualBlock(hidden_units, hidden_units),
            nn.Dropout(0.3),
            nn.Linear(hidden_units, 1),
        )

    def forward(self, x):
        return self.model(x)
