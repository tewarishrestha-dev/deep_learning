import torch
import matplotlib.pyplot as plt

from dataset import test_dataset, class_names
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
print("CNN FEATURE MAP VISUALIZATION")
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


# ============================================================
# Select an image
# ============================================================

image, label = test_dataset[0]

input_image = image.unsqueeze(0).to(DEVICE)

print()
print("Image shape:", image.shape)
print("True class:", class_names[label])


# ============================================================
# Forward through convolution layers
# ============================================================

with torch.no_grad():

    # First convolution block
    x = model.conv1(input_image)
    x = model.relu1(x)

    feature_map_1 = x

    x = model.pool1(x)

    # Second convolution block
    x = model.conv2(x)
    x = model.relu2(x)

    feature_map_2 = x


# ============================================================
# Print shapes
# ============================================================

print()
print("Feature map shapes:")
print("Conv1:", feature_map_1.shape)
print("Conv2:", feature_map_2.shape)


# ============================================================
# Visualize Conv1
# ============================================================

fig, axes = plt.subplots(
    2,
    4,
    figsize=(10, 5)
)

axes = axes.flatten()

for i in range(8):

    axes[i].imshow(
        feature_map_1[0, i].cpu(),
        cmap="gray"
    )

    axes[i].set_title(
        f"Conv1 Filter {i + 1}"
    )

    axes[i].axis("off")


plt.suptitle(
    f"Conv1 Feature Maps — "
    f"True: {class_names[label]}"
)

plt.tight_layout()

plt.savefig(
    "images/conv1_feature_maps.png",
    dpi=300
)

plt.show()


# ============================================================
# Visualize Conv2
# ============================================================

fig, axes = plt.subplots(
    4,
    4,
    figsize=(10, 10)
)

axes = axes.flatten()

for i in range(16):

    axes[i].imshow(
        feature_map_2[0, i].cpu(),
        cmap="gray"
    )

    axes[i].set_title(
        f"Conv2 Filter {i + 1}"
    )

    axes[i].axis("off")


plt.suptitle(
    f"Conv2 Feature Maps — "
    f"True: {class_names[label]}"
)

plt.tight_layout()

plt.savefig(
    "images/conv2_feature_maps.png",
    dpi=300
)

plt.show()


# ============================================================
# Complete
# ============================================================

print()
print("Saved:")
print("images/conv1_feature_maps.png")
print("images/conv2_feature_maps.png")

print()
print("=" * 60)
print("FEATURE MAP VISUALIZATION COMPLETE")
print("=" * 60)