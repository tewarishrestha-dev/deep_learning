import numpy as np

from flatten import Flatten


# ============================================================
# Test Flatten
# ============================================================

X = np.random.randn(
    1, 7, 7, 16
)

print("=" * 60)
print("FLATTEN TEST")
print("=" * 60)

print("Input shape:", X.shape)


# ------------------------------------------------------------
# Forward
# ------------------------------------------------------------

flatten = Flatten()

output = flatten.forward(X)

print("\nForward:")
print("Output shape:", output.shape)

print("Expected shape:", (1, 784))


# ------------------------------------------------------------
# Backward
# ------------------------------------------------------------

d_output = np.ones_like(output)

dX = flatten.backward(d_output)

print("\nBackward:")
print("Gradient shape:", dX.shape)

print("Expected shape:", X.shape)


# ------------------------------------------------------------
# Validation
# ------------------------------------------------------------

print("\n" + "=" * 60)

if output.shape == (1, 784):

    print("✓ Forward shape correct")

else:

    print("✗ Forward shape incorrect")


if dX.shape == X.shape:

    print("✓ Backward shape correct")

else:

    print("✗ Backward shape incorrect")


print("=" * 60)