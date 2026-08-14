import numpy as np
from cnn import CNN

print("=" * 60)
print("FULL CNN FORWARD + BACKWARD TEST")
print("=" * 60)

np.random.seed(42)

X = np.random.randn(1, 28, 28, 1)

print("\nInput:", X.shape)

model = CNN()

# Forward
output = model.forward(X)

print("Output:", output.shape)
print("Expected: (1, 10)")

# Backward
d_output = np.random.randn(1, 10)

dX = model.backward(d_output)

print("\nBackward:")
print("dX:", dX.shape)
print("Expected:", X.shape)

print("\n" + "=" * 60)
print("VALIDATION")
print("=" * 60)

if output.shape == (1, 10):
    print("✓ Forward shape correct")
else:
    print("✗ Forward shape incorrect")

if dX.shape == X.shape:
    print("✓ Backward shape correct")
else:
    print("✗ Backward shape incorrect")

print("=" * 60)