import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"

TRAIN_PATH = DATA_DIR / "fashion-mnist_train.csv"
TEST_PATH = DATA_DIR / "fashion-mnist_test.csv"


# --------------------------------------------------
# Load CSV
# --------------------------------------------------

print("Loading dataset...")

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

print("Dataset loaded.")


# --------------------------------------------------
# Separate features and labels
# --------------------------------------------------

y_train = train["label"].to_numpy()
y_test = test["label"].to_numpy()

X_train = train.drop(columns=["label"]).to_numpy()
X_test = test.drop(columns=["label"]).to_numpy()


print("\nBefore preprocessing:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
print("X_test :", X_test.shape)
print("y_test :", y_test.shape)


# --------------------------------------------------
# Normalize pixels
# --------------------------------------------------

X_train = X_train.astype(np.float32) / 255.0
X_test = X_test.astype(np.float32) / 255.0


# --------------------------------------------------
# Reshape images
# --------------------------------------------------

X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)


print("\nAfter preprocessing:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
print("X_test :", X_test.shape)
print("y_test :", y_test.shape)


# --------------------------------------------------
# Check values
# --------------------------------------------------

print("\nPixel range:")
print("Minimum:", X_train.min())
print("Maximum:", X_train.max())


# --------------------------------------------------
# Class names
# --------------------------------------------------

class_names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]


# --------------------------------------------------
# Example
# --------------------------------------------------

print("\nFirst image:")
print("Shape :", X_train[0].shape)
print("Label :", y_train[0])
print("Class :", class_names[y_train[0]])