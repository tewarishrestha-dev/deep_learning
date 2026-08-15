import os

import torch
import matplotlib.pyplot as plt

from dataset import test_dataset, class_names
from model_bn import CNN


# ============================================================
# Configuration
# ============================================================

MODEL_PATH = "models/best_model_bn.pth"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# Device
# ============================================================

print("=" * 60)
print("MISCLASSIFIED IMAGES - CNN + BATCHNORM")
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


checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()


# ============================================================
# Find misclassified images
# ============================================================

misclassified = []


with torch.no_grad():

    for index in range(len(test_dataset)):

        image, label = test_dataset[index]

        input_image = image.unsqueeze(0).to(
            DEVICE
        )

        output = model(
            input_image
        )

        prediction = output.argmax(
            dim=1
        ).item()

        if prediction != label:

            misclassified.append(
                (
                    image,
                    label,
                    prediction
                )
            )


# ============================================================
# Results
# ============================================================

print(
    "\nTotal misclassified:",
    len(misclassified)
)


# ============================================================
# Display first 12
# ============================================================

num_images = min(
    12,
    len(misclassified)
)

fig, axes = plt.subplots(
    3,
    4,
    figsize=(10, 8)
)

axes = axes.flatten()


for i in range(num_images):

    image, true_label, predicted_label = (
        misclassified[i]
    )

    axes[i].imshow(
        image.squeeze(),
        cmap="gray"
    )

    axes[i].set_title(
        f"True: {class_names[true_label]}\n"
        f"Pred: {class_names[predicted_label]}"
    )

    axes[i].axis("off")


# Hide unused axes

for i in range(
    num_images,
    len(axes)
):

    axes[i].axis("off")


plt.suptitle(
    "Fashion-MNIST CNN + BatchNorm\n"
    "Misclassified Images",
    fontsize=14
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
    "images/misclassified_images_bn.png"
)

plt.savefig(
    OUTPUT_PATH,
    dpi=300
)

plt.show()


print(
    "\nSaved to:"
)

print(OUTPUT_PATH)