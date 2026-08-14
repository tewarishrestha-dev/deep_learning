import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# PATH SETUP
# ============================================================

BASE_DIR = Path(__file__).parent
sys.path.append(str(BASE_DIR))


# ============================================================
# IMPORTS
# ============================================================

from load_data import X_train, y_train, class_names
from conv import Conv2D
from relu import ReLU
from pooling import MaxPool2D


# ============================================================
# 1. LOAD ONE REAL IMAGE
# ============================================================

X = X_train[0:1]

print("=" * 60)
print("INPUT IMAGE")
print("=" * 60)

print("Input shape:", X.shape)
print("Label:", y_train[0])
print("Class:", class_names[y_train[0]])


# ============================================================
# 2. CREATE CONVOLUTION LAYER
# ============================================================

conv = Conv2D(
    input_channels=1,
    num_filters=8,
    kernel_size=3,
    stride=1,
    padding=1
)

print("\n" + "=" * 60)
print("CONVOLUTION LAYER")
print("=" * 60)

print("Filter shape:", conv.W.shape)
print("Bias shape:", conv.b.shape)


# ============================================================
# 3. CONVOLUTION FORWARD
# ============================================================

output = conv.forward(X)

print("\n" + "=" * 60)
print("CONVOLUTION FORWARD")
print("=" * 60)

print("Input shape :", X.shape)
print("Output shape:", output.shape)

print("Output min:", output.min())
print("Output max:", output.max())


# ============================================================
# 4. SAVE CONVOLUTION FEATURE MAPS
# ============================================================

IMAGE_DIR = BASE_DIR.parent / "images"
IMAGE_DIR.mkdir(exist_ok=True)

plt.figure(figsize=(12, 6))

for i in range(8):

    plt.subplot(2, 4, i + 1)

    plt.imshow(
        output[0, :, :, i],
        cmap="gray"
    )

    plt.title(f"Filter {i + 1}")
    plt.axis("off")


plt.tight_layout()

conv_path = IMAGE_DIR / "convolution_output.png"

plt.savefig(
    conv_path,
    dpi=150,
    bbox_inches="tight"
)

plt.close()

print("\nConvolution feature maps saved to:")
print(conv_path)


# ============================================================
# 5. RELU FORWARD
# ============================================================

relu = ReLU()

relu_output = relu.forward(output)

print("\n" + "=" * 60)
print("RELU FORWARD")
print("=" * 60)

print("Before ReLU:")
print("Min:", output.min())
print("Max:", output.max())

print("\nAfter ReLU:")
print("Min:", relu_output.min())
print("Max:", relu_output.max())

print(
    "\nNegative values after ReLU:",
    np.sum(relu_output < 0)
)


# ============================================================
# 6. MAX POOLING FORWARD
# ============================================================

pool = MaxPool2D(
    pool_size=2,
    stride=2
)

pool_output = pool.forward(relu_output)

print("\n" + "=" * 60)
print("MAX POOLING FORWARD")
print("=" * 60)

print("Input shape :", relu_output.shape)
print("Output shape:", pool_output.shape)

print("Minimum:", pool_output.min())
print("Maximum:", pool_output.max())

# ============================================================
# SECOND CONVOLUTIONAL BLOCK
# ============================================================

conv2 = Conv2D(
    input_channels=8,
    num_filters=16,
    kernel_size=3,
    stride=1,
    padding=1
)

relu2 = ReLU()

pool2 = MaxPool2D(
    pool_size=2,
    stride=2
)


# ------------------------------------------------------------
# Conv2 forward
# ------------------------------------------------------------

output2 = conv2.forward(pool_output)

print("\n" + "=" * 60)
print("SECOND CONVOLUTION")
print("=" * 60)

print("Input shape :", pool_output.shape)
print("Output shape:", output2.shape)
print("Filter shape:", conv2.W.shape)


# ------------------------------------------------------------
# ReLU
# ------------------------------------------------------------

relu2_output = relu2.forward(output2)

print("\n" + "=" * 60)
print("SECOND RELU")
print("=" * 60)

print("Output shape:", relu2_output.shape)


# ------------------------------------------------------------
# MaxPool
# ------------------------------------------------------------

pool2_output = pool2.forward(relu2_output)

print("\n" + "=" * 60)
print("SECOND MAX POOL")
print("=" * 60)

print("Input shape :", relu2_output.shape)
print("Output shape:", pool2_output.shape)

# ============================================================
# SECOND CNN BLOCK - BACKWARD
# ============================================================

print("\n" + "=" * 60)
print("SECOND CNN BLOCK - BACKWARD")
print("=" * 60)


# Fake gradient coming from the next layer
d_pool2 = np.ones_like(pool2_output)


# Pool2 backward
d_relu2 = pool2.backward(d_pool2)

print("\nPool2 backward:")
print("Shape:", d_relu2.shape)
print("Expected:", relu2_output.shape)


# ReLU2 backward
d_conv2 = relu2.backward(d_relu2)

print("\nReLU2 backward:")
print("Shape:", d_conv2.shape)
print("Expected:", output2.shape)


# Conv2 backward
d_pool1 = conv2.backward(d_conv2)

print("\nConv2 backward:")
print("dX shape:", d_pool1.shape)
print("dW shape:", conv2.dW.shape)
print("db shape:", conv2.db.shape)

print("\nExpected:")
print("dX shape:", pool_output.shape)
print("dW shape:", conv2.W.shape)
print("db shape:", conv2.b.shape)


# ============================================================
# BACKWARD THROUGH FIRST BLOCK
# ============================================================

d_relu1 = pool.backward(d_pool1)

print("\nPool1 backward:")
print("Shape:", d_relu1.shape)
print("Expected:", relu_output.shape)


d_conv1 = relu.backward(d_relu1)

print("\nReLU1 backward:")
print("Shape:", d_conv1.shape)
print("Expected:", output.shape)


d_input = conv.backward(d_conv1)

print("\nConv1 backward:")
print("dX shape:", d_input.shape)
print("dW shape:", conv.dW.shape)
print("db shape:", conv.db.shape)

print("\nExpected:")
print("dX shape:", X.shape)
print("dW shape:", conv.W.shape)
print("db shape:", conv.b.shape)


# ============================================================
# 7. SAVE POOLING FEATURE MAPS
# ============================================================

plt.figure(figsize=(12, 6))

for i in range(8):

    plt.subplot(2, 4, i + 1)

    plt.imshow(
        pool_output[0, :, :, i],
        cmap="gray"
    )

    plt.title(f"Pool {i + 1}")
    plt.axis("off")


plt.tight_layout()

pool_path = IMAGE_DIR / "pooling_output.png"

plt.savefig(
    pool_path,
    dpi=150,
    bbox_inches="tight"
)

plt.close()

print("\nPooling feature maps saved to:")
print(pool_path)


# ============================================================
# 8. MAX POOLING BACKWARD
# ============================================================

# Simulate a gradient coming from the next layer.
#
# We use ones here only to test whether the backward
# propagation produces the correct shape.

d_pool = np.ones_like(pool_output)

d_relu = pool.backward(d_pool)

print("\n" + "=" * 60)
print("MAX POOLING BACKWARD")
print("=" * 60)

print("Incoming gradient shape :", d_pool.shape)
print("Outgoing gradient shape:", d_relu.shape)

print("Expected shape:", relu_output.shape)

print("Gradient min:", d_relu.min())
print("Gradient max:", d_relu.max())

print(
    "Non-zero gradients:",
    np.count_nonzero(d_relu)
)


# ============================================================
# 9. RELU BACKWARD
# ============================================================

d_conv = relu.backward(d_relu)

print("\n" + "=" * 60)
print("RELU BACKWARD")
print("=" * 60)

print("Incoming gradient shape :", d_relu.shape)
print("Outgoing gradient shape:", d_conv.shape)

print("Expected shape:", output.shape)

print("Gradient min:", d_conv.min())
print("Gradient max:", d_conv.max())


# ============================================================
# 10. CONVOLUTION BACKWARD
# ============================================================

dX = conv.backward(d_conv)

print("\n" + "=" * 60)
print("CONVOLUTION BACKWARD")
print("=" * 60)

print("dX shape:", dX.shape)
print("dW shape:", conv.dW.shape)
print("db shape:", conv.db.shape)

print("\nExpected:")
print("dX shape:", X.shape)
print("dW shape:", conv.W.shape)
print("db shape:", conv.b.shape)

print("\ndX min:", dX.min())
print("dX max:", dX.max())

print("dW min:", conv.dW.min())
print("dW max:", conv.dW.max())

print("db min:", conv.db.min())
print("db max:", conv.db.max())


# ============================================================
# 11. SHAPE VALIDATION
# ============================================================

print("\n" + "=" * 60)
print("SHAPE VALIDATION")
print("=" * 60)

checks = [

    (
        "Conv output",
        output.shape,
        (1, 28, 28, 8)
    ),

    (
        "ReLU output",
        relu_output.shape,
        (1, 28, 28, 8)
    ),

    (
        "Pool output",
        pool_output.shape,
        (1, 14, 14, 8)
    ),

    (
        "Pool backward",
        d_relu.shape,
        (1, 28, 28, 8)
    ),

    (
        "ReLU backward",
        d_conv.shape,
        (1, 28, 28, 8)
    ),

    (
        "Conv dX",
        dX.shape,
        (1, 28, 28, 1)
    ),

    (
        "Conv dW",
        conv.dW.shape,
        (8, 3, 3, 1)
    ),

    (
        "Conv db",
        conv.db.shape,
        (8,)
    )
]


all_passed = True

for name, actual, expected in checks:

    if actual == expected:

        print(f"✓ {name}: {actual}")

    else:

        print(
            f"✗ {name}: "
            f"got {actual}, expected {expected}"
        )

        all_passed = False


# ============================================================
# 12. FINAL RESULT
# ============================================================

print("\n" + "=" * 60)
print("TEST RESULT")
print("=" * 60)

if all_passed:

    print("✓ ALL SHAPE TESTS PASSED")

else:

    print("✗ SOME TESTS FAILED")