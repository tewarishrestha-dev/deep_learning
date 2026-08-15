import os

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix

from dataset import test_loader, class_names
from model import CNN


# ============================================================
# Configuration
# ============================================================

MODEL_PATH = "models/best_model.pth"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# Device
# ============================================================

print("=" * 60)
print("FASHION-MNIST CNN EVALUATION")
print("=" * 60)

print("Device:", DEVICE)

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
).to(DEVICE)


# ============================================================
# Load checkpoint
# ============================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("\nModel loaded:", MODEL_PATH)
print(
    "Saved epoch:",
    checkpoint["epoch"]
)
print(
    "Saved test accuracy:",
    f"{checkpoint['test_accuracy']:.4f}"
)


# ============================================================
# Loss
# ============================================================

criterion = nn.CrossEntropyLoss()


# ============================================================
# Evaluation
# ============================================================

correct = 0
total = 0
total_loss = 0.0

all_predictions = []
all_labels = []


with torch.no_grad():

    for X, y in test_loader:

        X = X.to(DEVICE)
        y = y.to(DEVICE)

        output = model(X)

        loss = criterion(
            output,
            y
        )

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

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_labels.extend(
            y.cpu().numpy()
        )


# ============================================================
# Results
# ============================================================

test_loss = total_loss / total
test_accuracy = correct / total

print("\n" + "=" * 60)
print("EVALUATION RESULTS")
print("=" * 60)

print(
    f"Test Loss:       {test_loss:.4f}"
)

print(
    f"Test Accuracy:   {test_accuracy:.4f}"
)

print(
    f"Correct:         {correct}/{total}"
)

print(
    f"Incorrect:       {total - correct}/{total}"
)


# ============================================================
# Confusion Matrix
# ============================================================

cm = confusion_matrix(
    all_labels,
    all_predictions
)

print("\nConfusion Matrix:")
print(cm)


# ============================================================
# Plot
# ============================================================

plt.figure(
    figsize=(10, 8)
)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.title(
    "Fashion-MNIST CNN Confusion Matrix"
)

plt.xticks(
    rotation=45,
    ha="right"
)

plt.yticks(
    rotation=0
)

plt.tight_layout()


# ============================================================
# Save
# ============================================================

os.makedirs(
    "images",
    exist_ok=True
)

OUTPUT_PATH = (
    "images/confusion_matrix.png"
)

plt.savefig(
    OUTPUT_PATH,
    dpi=300
)

plt.show()

print(
    f"\nConfusion matrix saved to:"
)

print(OUTPUT_PATH)

print("\n" + "=" * 60)
print("EVALUATION COMPLETE")
print("=" * 60)