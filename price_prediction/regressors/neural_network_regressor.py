import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from .base_regressor import BaseRegressor


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
        self.criterion = nn.MSELoss()
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

    def clear_fit(self):
        print("`clear_fit()` does nothing for now.")

    def train(self, X_train, y_train):
        # Ensure input size matches the data
        self.input_size = X_train.shape[1]
        self.model = self._build_model()
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=self.lr_factor, patience=self.lr_patience
        )

        # Scale the input data
        X_train = self.scaler.fit_transform(X_train)
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_train_tensor = (
            torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(self.device)
        )
        dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )

        # Early stopping variables
        best_loss = float("inf")
        epochs_no_improve = 0

        # Training loop
        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for batch_X, batch_y in dataloader:
                self.optimizer.zero_grad()  # Reset gradients
                outputs = self.model(batch_X)  # Forward pass
                loss = self.criterion(outputs, batch_y)  # Compute loss
                loss.backward()  # Backward pass
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=5.0
                )  # Clip gradients
                self.optimizer.step()  # Update weights
                epoch_loss += loss.item()

            # Average loss for the epoch
            epoch_loss /= len(dataloader)

            # Log epoch loss
            print(f"Epoch [{epoch + 1}/{self.epochs}], Loss: {epoch_loss:.4f}")

            # Check for improvement
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            # Reduce learning rate if loss plateaus
            prev_lr = self.optimizer.param_groups[0]["lr"]
            self.scheduler.step(epoch_loss)
            new_lr = self.optimizer.param_groups[0]["lr"]
            if new_lr != prev_lr:
                print(f"Learning rate reduced to {new_lr:.6f}")

            # Early stopping
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
