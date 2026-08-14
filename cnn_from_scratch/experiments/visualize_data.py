import matplotlib.pyplot as plt
from pathlib import Path

from load_data import X_train, y_train, class_names


# --------------------------------------------------
# Image output directory
# --------------------------------------------------

BASE_DIR = Path(__file__).parent
IMAGE_DIR = BASE_DIR / "images"

IMAGE_DIR.mkdir(exist_ok=True)


# --------------------------------------------------
# Visualize samples
# --------------------------------------------------

plt.figure(figsize=(8, 8))

for i in range(9):

    plt.subplot(3, 3, i + 1)

    plt.imshow(X_train[i].squeeze(), cmap="gray")

    plt.title(class_names[y_train[i]])

    plt.axis("off")


plt.tight_layout()


# --------------------------------------------------
# Save image
# --------------------------------------------------

output_path = IMAGE_DIR / "dataset_samples.png"

plt.savefig(output_path, dpi=150, bbox_inches="tight")

print(f"Image saved to: {output_path}")


# Show image
plt.show()