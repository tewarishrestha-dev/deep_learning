import torch
import matplotlib.pyplot as plt

from dataset import test_loader, class_names
from model import CNN


# ============================================================
# Configuration
# ============================================================

MODEL_PATH = "models/best_model.pth"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

NUM_IMAGES = 12


# ============================================================
# Device
# ============================================================

print("=" * 60)
print("MISCLASSIFIED IMAGE ANALYSIS")
print("=" * 60)

print("Device:", DEVICE)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# Load model
# ============================================================

model = CNN(
    num_classes=10
).to(DEVICE)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print(
    f"Model accuracy: "
    f"{checkpoint['test_accuracy']:.4f}"
)


# ============================================================
# Find incorrect predictions
# ============================================================

wrong_images = []
wrong_labels = []
wrong_predictions = []


with torch.no_grad():

    for X, y in test_loader:

        X = X.to(DEVICE)
        y = y.to(DEVICE)

        output = model(X)

        predictions = output.argmax(
            dim=1
        )

        incorrect = predictions != y

        for i in range(X.size(0)):

            if incorrect[i]:

                wrong_images.append(
                    X[i].cpu()
                )

                wrong_labels.append(
                    y[i].item()
                )

                wrong_predictions.append(
                    predictions[i].item()
                )

            if len(wrong_images) >= NUM_IMAGES:
                break

        if len(wrong_images) >= NUM_IMAGES:
            break


# ============================================================
# Print predictions
# ============================================================

print()
print("Misclassified Images:")

for i in range(len(wrong_images)):

    true_class = class_names[
        wrong_labels[i]
    ]

    predicted_class = class_names[
        wrong_predictions[i]
    ]

    print(
        f"{i + 1:02d}. "
        f"True: {true_class:<15} "
        f"Predicted: {predicted_class}"
    )


# ============================================================
# Visualization
# ============================================================

fig, axes = plt.subplots(
    3,
    4,
    figsize=(10, 8)
)

axes = axes.flatten()


for i, ax in enumerate(axes):

    image = wrong_images[i].squeeze()

    ax.imshow(
        image,
        cmap="gray"
    )

    true_class = class_names[
        wrong_labels[i]
    ]

    predicted_class = class_names[
        wrong_predictions[i]
    ]

    ax.set_title(
        f"True: {true_class}\n"
        f"Pred: {predicted_class}"
    )

    ax.axis("off")


plt.suptitle(
    "CNN Misclassified Fashion-MNIST Images",
    fontsize=14
)

plt.tight_layout()


# ============================================================
# Save
# ============================================================

plt.savefig(
    "images/misclassified_images.png",
    dpi=300
)

plt.show()

print()
print(
    "Saved to:"
)

print(
    "images/misclassified_images.png"
)

print()
print("=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)