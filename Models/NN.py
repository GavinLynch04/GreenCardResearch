import math

import torch
import torch.nn as nn
import torch.optim as optim
from skimage.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
import numpy as np
from Data.Preprocessing.preprocess import preprocess

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f'Using device: {device}')

X_raw, y_raw = preprocess()

X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
    X_raw.to_numpy(), y_raw.to_numpy(), test_size=0.25, random_state=27
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled = scaler.transform(X_test_raw) # Use transform, not fit_transform

# Convert to tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train_raw, dtype=torch.float32).unsqueeze(1).to(device) # Add dimension for MSELoss
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test_raw, dtype=torch.float32).unsqueeze(1).to(device) # Add dimension for MSELoss


# Define the neural network for Regression
class RegressionNN(nn.Module):
    def __init__(self, input_size):
        super(RegressionNN, self).__init__()
        # Keeping a similar architecture, adjust if needed
        self.fc1 = nn.Linear(input_size, 256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2) # Dropout can still be useful
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(32)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 1)  # Output 1 continuous value

    def forward(self, x):
        # Ensure batch norm is applied correctly even for single output regression
        # Handle potential edge case where batch size is 1 during eval
        if x.shape[0] > 1:
            x = self.relu(self.bn1(self.fc1(x)))
            x = self.dropout(x)
            x = self.relu(self.bn2(self.fc2(x)))
            x = self.dropout(x)
            x = self.relu(self.bn3(self.fc3(x)))
            x = self.dropout(x)
            x = self.relu(self.bn4(self.fc4(x)))
            x = self.dropout(x)
        else: # Skip BatchNorm if batch size is 1
             x = self.relu(self.fc1(x))
             x = self.dropout(x)
             x = self.relu(self.fc2(x))
             x = self.dropout(x)
             x = self.relu(self.fc3(x))
             x = self.dropout(x)
             x = self.relu(self.fc4(x))
             x = self.dropout(x)
        x = self.fc5(x) # No activation for regression output layer
        return x


# Model, loss, and optimizer
input_size = X_train_tensor.shape[1]

# Initialize the model
model = RegressionNN(input_size).to(device)

# Loss function and optimizer for Regression
criterion = nn.MSELoss()  # Mean Squared Error Loss for regression
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training parameters
epochs = 20 # Regression might need more epochs
batch_size = 64 # Adjusted batch size

# Training loop
print("Starting Training...")
for epoch in range(epochs):
    model.train()
    permutation = torch.randperm(X_train_tensor.size(0))

    running_loss = 0.0
    for i in range(0, X_train_tensor.size(0), batch_size):
        indices = permutation[i:i + batch_size]
        batch_X, batch_y = X_train_tensor[indices], y_train_tensor[indices]

        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y) # Target is float, shape [batch_size, 1]
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * batch_X.size(0)

    epoch_loss = running_loss / X_train_tensor.size(0)
    print(f"Epoch {epoch + 1}/{epochs}, Loss (MSE): {epoch_loss:.4f}")

# Test loop
print("\nStarting Evaluation...")
model.eval()
all_preds = []
all_targets = []
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    test_loss = criterion(test_outputs, y_test_tensor)

    # Store predictions and targets for sklearn metrics
    all_preds.extend(test_outputs.cpu().numpy())
    all_targets.extend(y_test_tensor.cpu().numpy())

# Calculate metrics using sklearn (more robust for final eval)
final_mse = mean_squared_error(all_targets, all_preds)
final_rmse = math.sqrt(final_mse)

# Alternatively, use the loss tensor directly (should be very similar)
# final_mse_tensor = test_loss.item()
# final_rmse_tensor = math.sqrt(final_mse_tensor)

print(f"\nTest Results (Train/Test Split):")
print(f"Test MSE: {final_mse:.4f}")
print(f"Test RMSE: {final_rmse:.4f}")


# --- Option 2: K-Fold Cross-Validation ---
print("\n--- Using K-Fold Cross-Validation ---")
k_folds = 5
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

# Standardize features within each fold
# Note: Scaling is done fold-wise here
fold_rmse_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_raw)):
    print(f"\nStarting Fold {fold + 1}/{k_folds}")

    # Split data for this fold
    X_train_fold, X_val_fold = X_raw[train_idx], X_raw[val_idx]
    y_train_fold, y_val_fold = y_raw[train_idx], y_raw[val_idx]

    # Scale features based on this fold's training data
    scaler_fold = StandardScaler()
    X_train_fold_scaled = scaler_fold.fit_transform(X_train_fold)
    X_val_fold_scaled = scaler_fold.transform(X_val_fold) # Use transform

    # Convert to tensors for this fold
    X_train_fold_tensor = torch.tensor(X_train_fold_scaled, dtype=torch.float32).to(device)
    y_train_fold_tensor = torch.tensor(y_train_fold, dtype=torch.float32).unsqueeze(1).to(device)
    X_val_fold_tensor = torch.tensor(X_val_fold_scaled, dtype=torch.float32).to(device)
    y_val_fold_tensor = torch.tensor(y_val_fold, dtype=torch.float32).unsqueeze(1).to(device)

    # Re-initialize model and optimizer for each fold!
    input_size_fold = X_train_fold_tensor.shape[1]
    model_fold = RegressionNN(input_size_fold).to(device)
    optimizer_fold = optim.Adam(model_fold.parameters(), lr=0.001)
    criterion_fold = nn.MSELoss()

    # Training loop for the fold
    for epoch in range(epochs): # Use the same number of epochs for simplicity
        model_fold.train()
        permutation = torch.randperm(X_train_fold_tensor.size(0))
        running_loss = 0.0
        for i in range(0, X_train_fold_tensor.size(0), batch_size):
            indices = permutation[i:i + batch_size]
            batch_X, batch_y = X_train_fold_tensor[indices], y_train_fold_tensor[indices]

            optimizer_fold.zero_grad()
            outputs = model_fold(batch_X)
            loss = criterion_fold(outputs, batch_y)
            loss.backward()
            optimizer_fold.step()
            running_loss += loss.item() * batch_X.size(0)
        # Optional: print epoch loss for the fold if desired
        # epoch_loss = running_loss / X_train_fold_tensor.size(0)
        # print(f" Fold {fold+1}, Epoch {epoch+1}, Loss: {epoch_loss:.4f}")


    # Validation for the fold
    model_fold.eval()
    fold_preds = []
    fold_targets = []
    with torch.no_grad():
        val_outputs = model_fold(X_val_fold_tensor)
        val_loss = criterion_fold(val_outputs, y_val_fold_tensor) # MSE Loss

        fold_preds.extend(val_outputs.cpu().numpy())
        fold_targets.extend(y_val_fold_tensor.cpu().numpy())

    fold_mse = mean_squared_error(fold_targets, fold_preds)
    fold_rmse = math.sqrt(fold_mse)
    fold_rmse_scores.append(fold_rmse)
    print(f"Fold {fold + 1} Validation RMSE: {fold_rmse:.4f}")

# Calculate and print average CV results
avg_rmse = np.mean(fold_rmse_scores)
std_rmse = np.std(fold_rmse_scores)
print(f"\nCross-Validation Results ({k_folds} Folds):")
print(f"Average RMSE: {avg_rmse:.4f}")
print(f"Standard Deviation RMSE: {std_rmse:.4f}")
