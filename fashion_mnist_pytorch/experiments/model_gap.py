import torch
import torch.nn as nn


class CNN(nn.Module):

    def __init__(self, num_classes=10):

        super().__init__()

        # ====================================================
        # Convolutional Block 1
        # ====================================================

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.bn1 = nn.BatchNorm2d(16)

        self.relu1 = nn.ReLU()

        self.pool1 = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )


        # ====================================================
        # Convolutional Block 2
        # ====================================================

        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.bn2 = nn.BatchNorm2d(32)

        self.relu2 = nn.ReLU()

        self.pool2 = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )


        # ====================================================
        # Global Average Pooling
        # ====================================================

        self.global_pool = nn.AdaptiveAvgPool2d(
            (1, 1)
        )


        # ====================================================
        # Classifier
        # ====================================================

        self.fc = nn.Linear(
            32,
            num_classes
        )


    # ========================================================
    # Forward
    # ========================================================

    def forward(self, x):

        # Conv block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        # Conv block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        # Global Average Pooling
        x = self.global_pool(x)

        # Flatten
        x = torch.flatten(
            x,
            start_dim=1
        )

        # Classification
        x = self.fc(x)

        return x