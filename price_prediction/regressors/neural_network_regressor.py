from sklearn.base import check_is_fitted
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler

from helpers.file_manager import FileManager
from .base_regressor import BaseRegressor


class MAPELoss(nn.Module):
    def forward(self, y_pred, y_true):
        epsilon = 1e-8  # To avoid division by zero
        return torch.mean(torch.abs((y_true - y_pred) / (y_true + epsilon))) * 100


class WideAndDeepModel(nn.Module):
    def __init__(self, input_size, hidden_units):
        super().__init__()
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


class NeuralNetworkRegressor(BaseRegressor):
    def __init__(
        self,
        hidden_units=128,
        learning_rate=0.001,
        epochs=100,
        batch_size=32,
        patience=10,  # Early stopping patience
        lr_patience=5,  # Learning rate reduction patience
        lr_factor=0.5,  # Learning rate reduction factor
    ):
        super().__init__()
        self.hidden_units = hidden_units
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.lr_patience = lr_patience
        self.lr_factor = lr_factor
        self.scaler = StandardScaler()
        # self.criterion = nn.MSELoss()
        self.criterion = MAPELoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _build_model(self):
        return nn.Sequential(
            nn.Linear(self.input_size, self.hidden_units),
            nn.ReLU(),
            nn.BatchNorm1d(self.hidden_units),
            nn.Dropout(0.3),
            nn.Linear(self.hidden_units, self.hidden_units),
            nn.ReLU(),
            nn.BatchNorm1d(self.hidden_units),
            nn.Dropout(0.3),
            nn.Linear(self.hidden_units, 1),
        )

    # def _build_model(self):
    #     return nn.Sequential(
    #         nn.Linear(self.input_size, self.hidden_units),
    #         nn.ReLU(),
    #         ResidualBlock(self.hidden_units, self.hidden_units),
    #         nn.Dropout(0.3),
    #         nn.Linear(self.hidden_units, 1),
    #     )

    # def _build_model(self):
    #     return WideAndDeepModel(self.input_size, self.hidden_units)

    def clear_fit(self):
        if hasattr(self, "model"):
            del self.model
            del self.optimizer
            del self.scheduler
            torch.cuda.empty_cache()

    def train(self, X_train, y_train, X_validation, y_validation):
        # Ensure input size matches the data
        self.input_size = X_train.shape[1]
        self.model = self._build_model()
        self.model.to(self.device)
        self.optimizer = Adam(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode="min", factor=self.lr_factor, patience=self.lr_patience
        )

        # Scale the input data
        X_train = self.scaler.fit_transform(X_train)
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_train_tensor = (
            torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(self.device)
        )
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_dataloader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )

        X_validation = self.scaler.transform(X_validation)
        X_val_tensor = torch.tensor(X_validation, dtype=torch.float32).to(self.device)
        y_val_tensor = (
            torch.tensor(y_validation.to_numpy(), dtype=torch.float32)
            .view(-1, 1)
            .to(self.device)
        )
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        val_dataloader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False
        )

        # Early stopping variables
        best_loss = float("inf")
        epochs_no_improve = 0
        tolerance = 1e-4

        # Training loop
        self.model.train()
        for epoch in range(self.epochs):
            print(f"Epoch {epoch + 1}/{self.epochs}\n" + "-" * 30)
            epoch_loss = 0.0
            for batch_X, batch_y in train_dataloader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                # clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                self.optimizer.step()  # Update weights
                epoch_loss += loss.item()

            epoch_loss /= len(train_dataloader)
            print(f"Training Loss: {epoch_loss:.4f}")

            # Validation loss
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_X, batch_y in val_dataloader:
                    outputs = self.model(batch_X)
                    loss = self.criterion(outputs, batch_y)
                    val_loss += loss.item()
            val_loss /= len(val_dataloader)
            print(f"Validation Loss: {val_loss:.4f}")

            # Check for improvement with tolerance
            if val_loss < best_loss - tolerance:
                best_loss = val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            # Reduce learning rate if loss plateaus
            prev_lr = self.optimizer.param_groups[0]["lr"]
            self.scheduler.step(val_loss)
            new_lr = self.optimizer.param_groups[0]["lr"]
            if new_lr != prev_lr:
                print(f"Learning rate reduced to {new_lr:.6f}")

            # Early stopping
            if epochs_no_improve >= self.patience:
                print(f"Early stopping triggered after {epoch + 1} epochs.")
                break

    def save_model(self, file_path: str) -> None:
        """
        Saves the trained model along with feature column names and types to a .pkl file.
        """
        data_to_save = {
            "model": self.model,
            "feature_columns": self.feature_columns,
            "feature_types": self.feature_types,
            "scaler_state": self.scaler,
        }
        FileManager.save_to_pkl(data_to_save, file_path)

    def load_model(self, file_path: str) -> bool:
        """
        Loads the model, feature columns, and feature types from a .pkl file.

        Returns:
            bool: True if loaded successfully, False otherwise.
        """
        loaded_data = FileManager.load_from_pkl(file_path)
        if loaded_data is not None:
            self.model = loaded_data.get("model")
            self.feature_columns = loaded_data.get("feature_columns")
            self.feature_types = loaded_data.get("feature_types")
            self.scaler = loaded_data.get("scaler_state")
            return True
        return False

    def predict(self, X_test):
        X_test = self.scaler.transform(X_test)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(self.device)
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_test_tensor).cpu().numpy()
        return predictions.flatten()
