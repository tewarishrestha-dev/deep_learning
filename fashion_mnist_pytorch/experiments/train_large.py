import os
import time

import torch
import torch.nn as nn
import torch.optim as optim

from dataset import train_loader, test_loader
from model_large import CNN


# ============================================================
# Configuration
# ============================================================

EPOCHS = 10
LEARNING_RATE = 0.001

MODEL_DIR = "models"
MODEL_PATH = os.path.join(
    MODEL_DIR,
    "best_model_large.pth"
)


# ============================================================
# Device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("LARGE CNN TRAINING")
print("=" * 60)

print("Device:", device)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# Model
# ============================================================

model = CNN(
    num_classes=10
).to(device)


# ============================================================
# Parameter Count
# ============================================================

total_parameters = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print(
    f"Total parameters: {total_parameters:,}"
)


# ============================================================
# Loss
# ============================================================

criterion = nn.CrossEntropyLoss()


# ============================================================
# Optimizer
# ============================================================

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# Create model directory
# ============================================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# ============================================================
# Training
# ============================================================

best_test_accuracy = 0.0

start_time = time.time()


for epoch in range(EPOCHS):

    # --------------------------------------------------------
    # Training mode
    # --------------------------------------------------------

    model.train()

    total_loss = 0.0
    correct = 0
    total = 0


    for X, y in train_loader:

        X = X.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        output = model(X)

        loss = criterion(
            output,
            y
        )

        loss.backward()

        optimizer.step()

        total_loss += (
            loss.item() * X.size(0)
        )

        predictions = output.argmax(
            dim=1
        )

        correct += (
            predictions == y
        ).sum().item()

        total += y.size(0)


    train_loss = total_loss / total
    train_accuracy = correct / total


    # ========================================================
    # Evaluation
    # ========================================================

    model.eval()

    correct = 0
    total = 0
    test_loss = 0.0


    with torch.no_grad():

        for X, y in test_loader:

            X = X.to(device)
            y = y.to(device)

            output = model(X)

            loss = criterion(
                output,
                y
            )

            test_loss += (
                loss.item() * X.size(0)
            )

            predictions = output.argmax(
                dim=1
            )

            correct += (
                predictions == y
            ).sum().item()

            total += y.size(0)


    test_loss /= total
    test_accuracy = correct / total


    # ========================================================
    # Print results
    # ========================================================

    print(
        f"\nEpoch [{epoch + 1}/{EPOCHS}]"
    )

    print(
        f"Train Loss:      {train_loss:.4f}"
    )

    print(
        f"Train Accuracy:  {train_accuracy:.4f}"
    )

    print(
        f"Test Loss:       {test_loss:.4f}"
    )

    print(
        f"Test Accuracy:   {test_accuracy:.4f}"
    )


    # ========================================================
    # Save best model
    # ========================================================

    if test_accuracy > best_test_accuracy:

        best_test_accuracy = test_accuracy

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "test_accuracy": test_accuracy,
                "epoch": epoch + 1
            },
            MODEL_PATH
        )

        print(
            f"✓ Best model saved "
            f"({test_accuracy:.4f})"
        )


# ============================================================
# Finished
# ============================================================

training_time = time.time() - start_time

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)

print(
    f"Best Test Accuracy: "
    f"{best_test_accuracy:.4f}"
)

print(
    f"Training Time: "
    f"{training_time:.2f} seconds"
)

print(
    f"Model saved to: "
    f"{MODEL_PATH}"
)