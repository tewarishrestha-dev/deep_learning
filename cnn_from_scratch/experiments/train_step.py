import numpy as np
from cnn import CNN
from loss import SoftmaxCrossEntropy

np.random.seed(42)

# One fake MNIST image
X = np.random.randn(1, 28, 28, 1)

# Digit = 3
y = np.array([[0, 0, 0, 1, 0, 0, 0, 0, 0, 0]])

model = CNN()
loss_fn = SoftmaxCrossEntropy()

lr = 0.001

# ============================================================
# FORWARD
# ============================================================

logits = model.forward(X)

loss = loss_fn.forward(logits, y)

print("Before update")
print("Loss:", loss)

# ============================================================
# BACKWARD
# ============================================================

d_logits = loss_fn.backward()

model.backward(d_logits)

print("\nGradient magnitudes:")
print("conv1 dW:", np.max(np.abs(model.conv1.dW)))
print("conv2 dW:", np.max(np.abs(model.conv2.dW)))
print("fc1 dW  :", np.max(np.abs(model.fc1.dW)))
print("fc2 dW  :", np.max(np.abs(model.fc2.dW)))

# ============================================================
# UPDATE
# ============================================================

model.conv1.W -= lr * model.conv1.dW
model.conv1.b -= lr * model.conv1.db

model.conv2.W -= lr * model.conv2.dW
model.conv2.b -= lr * model.conv2.db

model.fc1.W -= lr * model.fc1.dW
model.fc1.b -= lr * model.fc1.db

model.fc2.W -= lr * model.fc2.dW
model.fc2.b -= lr * model.fc2.db

# ============================================================
# FORWARD AGAIN
# ============================================================

logits = model.forward(X)

loss_after = loss_fn.forward(logits, y)

print("\nAfter update")
print("Loss:", loss_after)

print("\nLoss change:", loss_after - loss)

if loss_after < loss:
    print("✓ Loss decreased")
else:
    print("✗ Loss did not decrease")