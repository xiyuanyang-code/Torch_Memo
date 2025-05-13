"""
Author: Xiyuan Yang   xiyuan_yang@outlook.com
Date: 2025-03-19 16:38:23
LastEditors: Xiyuan Yang   xiyuan_yang@outlook.com
LastEditTime: 2025-03-19 20:43:05
FilePath: /pytorch-deep-learning/exercise/exercise1.py
Description:
Do you code and make progress today?
Copyright (c) 2025 by Xiyuan Yang, All Rights Reserved.
"""

# The initial Plotting image for Model Training (Linear Regression)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# Set random seed for reproducibility
torch.manual_seed(2025)

# Font settings
font_georgia = FontProperties(
    fname="/GPFS/rhome/xiyuanyang/.local/share/fonts/georgia.ttf"
)
font_songti = FontProperties(
    fname="/GPFS/rhome/xiyuanyang/.local/share/fonts/songti.ttc"
)

# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"

# Data generation
weight = 0.3
bias = 0.9
X = torch.arange(0, 5, 0.01).unsqueeze(dim=1)
y = weight * X + bias + torch.rand((X.shape[0], 1)) * 0.3

# Split the data
n_samples = X.shape[0]
train_size = int(0.7 * n_samples)
test_size = n_samples - train_size
indices = torch.randperm(n_samples)
X_train, y_train = X[indices[:train_size]], y[indices[:train_size]]
X_test, y_test = X[indices[train_size:]], y[indices[train_size:]]

# Plot split data
plt.figure(figsize=(10, 6))
plt.scatter(x=X_train, y=y_train, color="red", label="Train data", alpha=0.3, s=5)
plt.scatter(x=X_test, y=y_test, color="blue", label="Test data", alpha=0.3, s=5)
plt.xlabel("X axis")
plt.ylabel("Y axis")
plt.legend()
plt.savefig("splitted_1.png")
plt.close()


# Define the model
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_layer = nn.Linear(in_features=1, out_features=1)

    def forward(self, x):
        return self.linear_layer(x)


# Initialize model, loss function, and optimizer
model = LinearRegressionModel().to(device)
loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)

# Move data to the same device as the model
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

# Training loop
epochs = 50000
train_losses, test_losses = [], []

for epoch in range(epochs):
    # Training phase
    model.train()
    y_pred = model(X_train)
    loss = loss_fn(y_pred, y_train)
    train_losses.append(loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Evaluation phase
    model.eval()
    with torch.inference_mode():
        test_pred = model(X_test)
        test_loss = loss_fn(test_pred, y_test)
        test_losses.append(test_loss.item())

    # Print progress every 100 epochs
    if epoch % 100 == 0:
        print(
            f"Epoch: {epoch} | Train Loss: {loss.item():.4f} | Test Loss: {test_loss.item():.4f}"
        )

# Plot training and testing losses
plt.figure(figsize=(10, 6))
plt.plot(range(epochs), train_losses, label="Train Loss", color="red", alpha=0.7)
plt.plot(range(epochs), test_losses, label="Test Loss", color="blue", alpha=0.7)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Testing Loss", fontproperties=font_georgia, fontsize=18)
plt.legend()
plt.savefig("loss_plot1.png")
plt.close()

# Print model parameters
from pprint import pprint

pprint(model.state_dict())
