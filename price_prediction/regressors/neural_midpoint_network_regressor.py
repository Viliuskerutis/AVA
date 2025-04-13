import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from config import REGRESSOR_NEURAL_MIDPOINT_NETWORK_PKL_PATH
from price_prediction.regressors.base_regressor import BaseRegressor
from price_prediction.regressors.neural_network.base_model import BaseModel
from price_prediction.regressors.neural_network.midpoint_deviation_loss import (
    MidpointDeviationLoss,
)


class NeuralMidpointNetworkRegressor(BaseRegressor):
    def __init__(
        self,
        model_class: BaseModel,
        loss_function=MidpointDeviationLoss(),
        hidden_units=128,
        learning_rate=0.001,
        epochs=100,
        batch_size=32,
        patience=10,
        lr_patience=5,
        lr_factor=0.5,
    ):
        super().__init__()
        self.model_class = model_class
        self.loss_function = loss_function
        self.hidden_units = hidden_units
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.lr_patience = lr_patience
        self.lr_factor = lr_factor
        self.scaler = StandardScaler()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.path = REGRESSOR_NEURAL_MIDPOINT_NETWORK_PKL_PATH

    def _build_model(self):
        return self.model_class(self.input_size, self.hidden_units)

    def clear_fit(self):
        if hasattr(self, "model"):
            del self.model
            del self.optimizer
            del self.scheduler
            torch.cuda.empty_cache()

    def train(
        self,
        X_train,
        y_train,
        X_validation,
        y_validation,
        midpoints_train,
        midpoints_validation,
    ):
        self.input_size = X_train.shape[1]
        self.model = self._build_model()
        self.model.to(self.device)
        self.optimizer = Adam(self.model.parameters(), lr=self.learning_rate)

        # Scale the input data
        X_train = self.scaler.fit_transform(X_train)
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        midpoints_train_tensor = (
            torch.tensor(midpoints_train.to_numpy(), dtype=torch.float32)
            .view(-1, 1)
            .to(self.device)
        )
        train_dataset = TensorDataset(X_train_tensor, midpoints_train_tensor)
        train_dataloader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )

        X_validation = self.scaler.transform(X_validation)
        X_val_tensor = torch.tensor(X_validation, dtype=torch.float32).to(self.device)
        midpoints_validation_tensor = (
            torch.tensor(midpoints_validation.to_numpy(), dtype=torch.float32)
            .view(-1, 1)
            .to(self.device)
        )

        # Training loop
        best_loss = float("inf")
        epochs_no_improve = 0
        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0.0
            for batch_X, batch_midpoints in train_dataloader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.loss_function(outputs, batch_midpoints)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            epoch_loss /= len(train_dataloader)
            print(f"Epoch {epoch + 1}/{self.epochs}, Training Loss: {epoch_loss:.4f}")

            # Validation loss
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val_tensor)
                val_loss = self.loss_function(
                    val_outputs, midpoints_validation_tensor
                ).item()
            print(f"Validation Loss: {val_loss:.4f}")

            if val_loss < best_loss:
                best_loss = val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= self.patience:
                print("Early stopping triggered.")
                break

    def predict(self, X_test):
        X_test = self.scaler.transform(X_test)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(self.device)
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_test_tensor).cpu().numpy()
        return predictions.flatten()
