"""
Author: Xiyuan Yang   xiyuan_yang@outlook.com
Date: 2025-04-15 14:40:20
LastEditors: Xiyuan Yang   xiyuan_yang@outlook.com
LastEditTime: 2025-04-15 14:41:31
FilePath: /CNN-tutorial/src/AlexNet.py
Description:
Do you code and make progress today?
Copyright (c) 2025 by Xiyuan Yang, All Rights Reserved.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AlexNet(nn.Module):
    def __init__(self, num_classes=10):  # CIFAR-10 has 10 classes
        super(AlexNet, self).__init__()

        # Define the convolutional layers
        self.features = nn.Sequential(
            nn.Conv2d(
                3, 64, kernel_size=3, stride=1, padding=1
            ),  # Adjusted for 32x32 input
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 16x16
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 8x8
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 4x4
        )

        # Define the fully connected layers
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 4 * 4, 4096),  # Adjusted for 4x4 feature map
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# Example usage
if __name__ == "__main__":
    # Create an instance of AlexNet with 10 output classes (CIFAR-10)
    model = AlexNet(num_classes=10)

    # Print the model architecture
    print(model)

    # Test with a random input tensor (batch size 1, 3 channels, 32x32 image)
    input_tensor = torch.randn(1, 3, 32, 32)
    output = model(input_tensor)
    print(output.shape)  # Should output torch.Size([1, 10])
