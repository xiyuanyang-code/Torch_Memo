# Kaggle Competition for SJTU Math1116
# Task: Train a ResNet on CIFAR-10 for image classification

print("Starting module imports")
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import matplotlib as mlp

mlp.use("Agg")
import time
import matplotlib.pyplot as plt
from torchvision.models import resnet18
import torchvision.transforms as transforms
from PIL import Image

# Get timestamp
timestamp = time.time()
print(f"Time: {timestamp}")
print("Imports finished!")

# Load data
print("Loading data...")
file_test_path = "./data/test_data.npz"
file_train_path = "./data/train_data.npz"
train_data = np.load(file_train_path)
test_data = np.load(file_test_path)

x_train, y_train = train_data["x_train"], train_data["y_train"]
x_test, test_ids = test_data["x_test"], np.arange(len(test_data["x_test"]))

# Convert to torch.tensor for training
x_train = torch.tensor(x_train / 255.0, dtype=torch.float32).permute(
    0, 2, 3, 1
)  # (N, H, W, C)
x_test = torch.tensor(x_test / 255.0, dtype=torch.float32).permute(0, 2, 3, 1)
y_train = torch.tensor(y_train, dtype=torch.long)
print("Data loaded successfully!")

# Data augmentation
train_transforms = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),  # Random horizontal flip
        transforms.RandomCrop(32, padding=4),  # Random crop with padding
        transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
        ),  # Color jitter
        transforms.RandomRotation(15),  # Random rotation
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalize
    ]
)

test_transforms = transforms.Compose(
    [
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalize
    ]
)


# Custom Dataset
class CIFARDataset(Dataset):
    def __init__(self, x, y=None, train=True, transform=None):
        self.x = x
        self.y = y
        self.train = train
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        # Get original image
        img = self.x[idx].permute(2, 0, 1).numpy().astype(np.uint8)  # (C, H, W)
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))  # (H, W, C)

        # Apply data augmentation
        if self.transform:
            augmented_img = self.transform(img)
        else:
            augmented_img = transforms.ToTensor()(img)

        # Convert original image to tensor
        original_img = transforms.ToTensor()(img)

        if self.train:
            return original_img, augmented_img, self.y[idx]
        else:
            return original_img, augmented_img


# Load DataLoader with Data Augmentation
print("Loading DataLoader with data augmentation")
train_dataset = CIFARDataset(x_train, y_train, train=True, transform=train_transforms)
test_dataset = CIFARDataset(x_test, train=False, transform=test_transforms)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
print("DataLoader loaded successfully!")


# Pretrained ResNet18
def PretrainedResNet18(num_classes=10):
    model = resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# Build Model
print("Building pretrained ResNet18 model")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PretrainedResNet18().to(device)
model = nn.DataParallel(model)
model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

# Training function
print("Starting training process")


def train_model(num_epochs):
    train_losses = []
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for _, augmented_inputs, labels in train_loader:
            inputs, labels = augmented_inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
        scheduler.step()

    # Plot training loss curve
    plt.plot(train_losses)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(f"fig/train_loss_{timestamp}.png")
    plt.close()


# Start training
train_model(num_epochs=30)


# Test function
def predict_test():
    model.eval()
    predictions = []
    with torch.no_grad():
        for _, augmented_inputs in test_loader:
            inputs = augmented_inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.cpu().numpy())
    return predictions


print("Training done, starting evaluation")
# Generate predictions
predictions = predict_test()

# Save as CSV file
submission_df = pd.DataFrame({"ID": test_ids, "Label": predictions})
submission_df.to_csv(f"answer/submission_{timestamp}.csv", index=False)
print("Predictions saved to submission.csv")
