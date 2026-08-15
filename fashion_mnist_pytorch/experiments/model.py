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
            out_channels=8,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.relu1 = nn.ReLU()

        self.pool1 = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )


        # ====================================================
        # Convolutional Block 2
        # ====================================================

        self.conv2 = nn.Conv2d(
            in_channels=8,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.relu2 = nn.ReLU()

        self.pool2 = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )


        # ====================================================
        # Classifier
        # ====================================================

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(
            7 * 7 * 16,
            128
        )

        self.relu3 = nn.ReLU()

        self.fc2 = nn.Linear(
            128,
            num_classes
        )


    # ========================================================
    # Forward
    # ========================================================

    def forward(self, x):

        # Conv block 1
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        # Conv block 2
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        # Classifier
        x = self.flatten(x)

        x = self.fc1(x)
        x = self.relu3(x)

        x = self.fc2(x)

        return x