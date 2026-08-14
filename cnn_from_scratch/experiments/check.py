import pandas as pd
import numpy as np
from pathlib import Path


# --------------------------------------------------
# Paths
# --------------------------------------------------

DATA_DIR = Path("data")

train_path = DATA_DIR / "fashion-mnist_train.csv"
test_path = DATA_DIR / "fashion-mnist_test.csv"


# --------------------------------------------------
# Load datasets
# --------------------------------------------------

print("=" * 60)
print("LOADING DATA")
print("=" * 60)

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

print("Train loaded successfully")
print("Test loaded successfully")


# --------------------------------------------------
# 1. Dataset shape
# --------------------------------------------------

print("\n" + "=" * 60)
print("1. DATASET SHAPE")
print("=" * 60)

print("Train shape:", train.shape)
print("Test shape :", test.shape)


# --------------------------------------------------
# 2. Expected dimensions
# --------------------------------------------------

print("\n" + "=" * 60)
print("2. EXPECTED DIMENSIONS")
print("=" * 60)

print("Expected train rows: 60000")
print("Actual train rows  :", len(train))

print("Expected test rows : 10000")
print("Actual test rows   :", len(test))

print("Expected columns   : 785")
print("Actual columns     :", train.shape[1])

print("\nExpected:")
print("1 label + 784 pixels")


# --------------------------------------------------
# 3. Column names
# --------------------------------------------------

print("\n" + "=" * 60)
print("3. COLUMN INFORMATION")
print("=" * 60)

print("First 10 columns:")
print(train.columns[:10].tolist())

print("\nLast 5 columns:")
print(train.columns[-5:].tolist())


# --------------------------------------------------
# 4. Missing values
# --------------------------------------------------

print("\n" + "=" * 60)
print("4. MISSING VALUES")
print("=" * 60)

train_missing = train.isnull().sum().sum()
test_missing = test.isnull().sum().sum()

print("Total missing values in train:", train_missing)
print("Total missing values in test :", test_missing)

if train_missing == 0 and test_missing == 0:
    print("✓ No missing values found")
else:
    print("⚠ Missing values found!")


# --------------------------------------------------
# 5. Check labels
# --------------------------------------------------

print("\n" + "=" * 60)
print("5. LABELS")
print("=" * 60)

print("Unique train labels:")
print(sorted(train["label"].unique()))

print("\nNumber of unique labels:")
print(train["label"].nunique())


# --------------------------------------------------
# 6. Label distribution
# --------------------------------------------------

print("\n" + "=" * 60)
print("6. LABEL DISTRIBUTION")
print("=" * 60)

print(train["label"].value_counts().sort_index())


# --------------------------------------------------
# 7. Pixel range
# --------------------------------------------------

print("\n" + "=" * 60)
print("7. PIXEL VALUES")
print("=" * 60)

X_train_pixels = train.drop(columns=["label"])

print("Minimum pixel value:", X_train_pixels.min().min())
print("Maximum pixel value:", X_train_pixels.max().max())

print("\nExpected range:")
print("0 → 255")


# --------------------------------------------------
# 8. Pixel data type
# --------------------------------------------------

print("\n" + "=" * 60)
print("8. PIXEL DATA TYPE")
print("=" * 60)

print(X_train_pixels.dtypes.value_counts())


# --------------------------------------------------
# 9. Check number of pixels
# --------------------------------------------------

print("\n" + "=" * 60)
print("9. NUMBER OF PIXELS")
print("=" * 60)

num_pixels = len(X_train_pixels.columns)

print("Pixels per image:", num_pixels)

if num_pixels == 784:
    print("✓ Correct: 784 pixels = 28 × 28")
else:
    print("⚠ Unexpected number of pixels")


# --------------------------------------------------
# 10. Check whether all pixel values are integers
# --------------------------------------------------

print("\n" + "=" * 60)
print("10. PIXEL VALUE VALIDATION")
print("=" * 60)

all_valid = (
    X_train_pixels.to_numpy().min() >= 0
    and X_train_pixels.to_numpy().max() <= 255
)

if all_valid:
    print("✓ All pixel values are between 0 and 255")
else:
    print("⚠ Invalid pixel values found!")


# --------------------------------------------------
# 11. Show first few rows
# --------------------------------------------------

print("\n" + "=" * 60)
print("11. FIRST 5 ROWS")
print("=" * 60)

print(train.head())


# --------------------------------------------------
# 12. Check first image
# --------------------------------------------------

print("\n" + "=" * 60)
print("12. FIRST IMAGE")
print("=" * 60)

first_image = X_train_pixels.iloc[0].to_numpy()

print("Flattened shape:", first_image.shape)

image = first_image.reshape(28, 28)

print("Reshaped shape:", image.shape)

print("\n28 × 28 image:")
print(image)


# --------------------------------------------------
# 13. Check image statistics
# --------------------------------------------------

print("\n" + "=" * 60)
print("13. FIRST IMAGE STATISTICS")
print("=" * 60)

print("Minimum:", image.min())
print("Maximum:", image.max())
print("Mean   :", image.mean())

print("\nLabel of first image:", train.iloc[0]["label"])


# --------------------------------------------------
# Final summary
# --------------------------------------------------

print("\n" + "=" * 60)
print("DATASET CHECK COMPLETE")
print("=" * 60)