import torch
import torch.nn as nn
import torchvision.models as models
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler

from config import DATA_PATH
from helpers.file_manager import FileManager
from price_prediction.regressors.base_regressor import BaseRegressor


REGRESSOR_DENSENET_PKL_PATH = f"{DATA_PATH}/densenet.pkl"


class DenseNetRegressorModel(nn.Module):
    def __init__(self, input_size, output_size=1):
        super().__init__()
        # Load pretrained DenseNet
        self.densenet = models.densenet121(pretrained=True)

        # self.densenet.features.denseblock1 = nn.Identity()
        # self.densenet.features.transition1 = nn.Identity()
        # self.densenet.features.denseblock2 = nn.Identity()
        # self.densenet.features.transition2 = nn.Identity()

        # Modify the first layer to accept tabular data input
        # DenseNet expects image data, so we need to reshape our input
        self.input_adapter = nn.Sequential(
            nn.Linear(input_size, 64 * 64 * 3), nn.ReLU(), nn.BatchNorm1d(64 * 64 * 3)
        )

        # Modify DenseNet's classifier for regression
        num_features = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, output_size),
        )

    def forward(self, x):
        # Reshape tabular data to image-like format
        x = self.input_adapter(x)
        x = x.view(
            -1, 3, 64, 64
        )  # Reshape to image format (batch_size, channels, height, width)

        # Process through DenseNet
        x = self.densenet(x)
        return x


class DenseNetRegressor(BaseRegressor):
    def __init__(
        self,
        loss_function=nn.L1Loss(),
        learning_rate=0.001,
        epochs=100,
        batch_size=32,
        patience=10,
        lr_patience=5,
        lr_factor=0.5,
    ):
        super().__init__()
        self.loss_function = loss_function
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.lr_patience = lr_patience
        self.lr_factor = lr_factor
        self.scaler = StandardScaler()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.path = REGRESSOR_DENSENET_PKL_PATH

    def clear_fit(self):
        if hasattr(self, "model"):
            del self.model
            del self.optimizer
            del self.scheduler
            torch.cuda.empty_cache()

    def train(self, X_train, y_train, X_validation=None, y_validation=None):
        # Ensure input size matches the data
        self.input_size = X_train.shape[1]
        self.model = DenseNetRegressorModel(self.input_size)
        self.model.to(self.device)

        # Define optimizer and scheduler
        self.optimizer = Adam(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode="min", factor=self.lr_factor, patience=self.lr_patience
        )

        # Scale the input data
        X_train = self.scaler.fit_transform(X_train)
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_train_tensor = (
            torch.tensor(y_train.to_numpy(), dtype=torch.float32)
            .view(-1, 1)
            .to(self.device)
        )

        # Create DataLoader
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_dataloader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )

        # Handle validation data if provided
        val_dataloader = None
        if X_validation is not None and y_validation is not None:
            X_validation = self.scaler.transform(X_validation)
            X_val_tensor = torch.tensor(X_validation, dtype=torch.float32).to(
                self.device
            )
            y_val_tensor = (
                torch.tensor(y_validation.to_numpy(), dtype=torch.float32)
                .view(-1, 1)
                .to(self.device)
            )
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            val_dataloader = DataLoader(
                val_dataset, batch_size=self.batch_size, shuffle=False
            )

        # Training loop
        best_loss = float("inf")
        epochs_no_improve = 0

        for epoch in range(self.epochs):
            print(f"Epoch {epoch + 1}/{self.epochs}\n" + "-" * 30)

            # Training phase
            self.model.train()
            epoch_loss = 0.0
            for batch_X, batch_y in train_dataloader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.loss_function(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            epoch_loss /= len(train_dataloader)
            print(f"Training Loss: {epoch_loss:.4f}")

            # Validation phase if validation data is provided
            if val_dataloader:
                self.model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch_X, batch_y in val_dataloader:
                        outputs = self.model(batch_X)
                        loss = self.loss_function(outputs, batch_y)
                        val_loss += loss.item()

                val_loss /= len(val_dataloader)
                print(f"Validation Loss: {val_loss:.4f}")

                # Update learning rate based on validation loss
                self.scheduler.step(val_loss)

            # Early stopping
            if val_loss < best_loss:
                best_loss = val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= self.patience:
                print(f"Early stopping triggered after {epoch + 1} epochs.")
                break

    def predict(self, X_test):
        X_test = self.scaler.transform(X_test)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(self.device)
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_test_tensor).cpu().numpy()
        return predictions.flatten()

    def save_model(self, file_path: str) -> None:
        data_to_save = {
            "model": self.model,
            "feature_columns": self.feature_columns,
            "feature_types": self.feature_types,
            "scaler_state": self.scaler,
        }
        FileManager.save_to_pkl(data_to_save, file_path)

    def load_model(self, file_path: str) -> bool:
        loaded_data = FileManager.load_from_pkl(file_path)
        if loaded_data is not None:
            self.model = loaded_data.get("model")
            self.feature_columns = loaded_data.get("feature_columns")
            self.feature_types = loaded_data.get("feature_types")
            self.scaler = loaded_data.get("scaler_state")
            return True
        return False
