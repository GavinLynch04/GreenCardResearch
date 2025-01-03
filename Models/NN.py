import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from Data.Preprocessing.preprocess import preprocess

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

X_encoded, y = preprocess()

category_mapping = {
    "0-2.5 years": 0,
    "2.5-5 years": 1,
    "5-10 years": 2,
    ">10 years": 3
}
y = np.vectorize(category_mapping.get)(y)
X_encoded = X_encoded.astype(int)

# Standardize features
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X_encoded)

X_tensor = torch.tensor(X_normalized, dtype=torch.float32).to(device)
y_tensor = torch.tensor(y, dtype=torch.float32).to(device)

# Split data into training and testing sets
train_X, test_X, train_y, test_y = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=27)


# Define the neural network
class ClassificationNN(nn.Module):
    def __init__(self, input_size):
        super(ClassificationNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(32)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 4)  # 4 output classes for 4-class classification

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        x = self.relu(self.bn4(self.fc4(x)))
        x = self.dropout(x)
        x = self.fc5(x)
        return x


# Model, loss, and optimizer
input_size = train_X.shape[1]
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, accuracy_score

# Initialize the model
model = ClassificationNN(input_size).to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()  # Use CrossEntropyLoss for multi-class classification
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training parameters
epochs = 5
batch_size = 100

# Training loop
for epoch in range(epochs):
    model.train()
    permutation = torch.randperm(train_X.size(0))  # Shuffle training data

    for i in range(0, train_X.size(0), batch_size):
        indices = permutation[i:i + batch_size]
        batch_X, batch_y = train_X[indices], train_y[indices]

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(batch_X)

        loss = criterion(outputs, batch_y.long())  # Ensure targets are long for CrossEntropyLoss

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

# Test loop
model.eval()
with torch.no_grad():
    test_outputs = model(test_X)
    _, predicted = torch.max(test_outputs, 1)  # Get class predictions
    test_loss = criterion(test_outputs, test_y.long())

    # Calculate metrics
    accuracy = accuracy_score(test_y.cpu(), predicted.cpu())
    precision = precision_score(test_y.cpu(), predicted.cpu(), average='weighted')
    recall = recall_score(test_y.cpu(), predicted.cpu(), average='weighted')
    f1 = f1_score(test_y.cpu(), predicted.cpu(), average='weighted')

    print(f"\nTest Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(classification_report(test_y.cpu(), predicted.cpu()))

