import numpy as np

from dense import Dense


np.random.seed(42)


# ============================================================
# Create test input
# ============================================================

X = np.random.randn(
    1, 784
)


# ============================================================
# Create Dense layer
# ============================================================

dense = Dense(
    input_size=784,
    output_size=128
)


print("=" * 60)
print("DENSE LAYER TEST")
print("=" * 60)

print("Input shape :", X.shape)
print("Weight shape:", dense.W.shape)
print("Bias shape  :", dense.b.shape)


# ============================================================
# Forward
# ============================================================

output = dense.forward(X)

print("\nForward:")
print("Output shape:", output.shape)

print("Expected shape:", (1, 128))


# ============================================================
# Backward
# ============================================================

d_output = np.random.randn(
    1, 128
)

dX = dense.backward(d_output)

print("\nBackward:")

print("dX shape:", dX.shape)
print("dW shape:", dense.dW.shape)
print("db shape:", dense.db.shape)

print("\nExpected:")

print("dX shape:", X.shape)
print("dW shape:", dense.W.shape)
print("db shape:", dense.b.shape)


# ============================================================
# Validation
# ============================================================

print("\n" + "=" * 60)
print("VALIDATION")
print("=" * 60)


if output.shape == (1, 128):
    print("✓ Forward shape correct")
else:
    print("✗ Forward shape incorrect")


if dX.shape == X.shape:
    print("✓ dX shape correct")
else:
    print("✗ dX shape incorrect")


if dense.dW.shape == dense.W.shape:
    print("✓ dW shape correct")
else:
    print("✗ dW shape incorrect")


if dense.db.shape == dense.b.shape:
    print("✓ db shape correct")
else:
    print("✗ db shape incorrect")


print("=" * 60)