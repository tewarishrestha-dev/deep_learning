import numpy as np
from loss import SoftmaxCrossEntropy

print("=" * 60)
print("SOFTMAX + CROSS ENTROPY TEST")
print("=" * 60)

np.random.seed(42)

logits = np.array([
    [2.0, 1.0, 0.1],
    [0.2, 0.3, 2.0]
])

y_true = np.array([
    [1, 0, 0],
    [0, 0, 1]
])

loss_fn = SoftmaxCrossEntropy()

loss = loss_fn.forward(logits, y_true)
d_logits = loss_fn.backward()

print("\nLogits:")
print(logits)

print("\nProbabilities:")
print(loss_fn.probs)

print("\nLoss:", loss)

print("\nd_logits:")
print(d_logits)

print("\nProbability sums:")
print(np.sum(loss_fn.probs, axis=1))

print("\n" + "=" * 60)
print("VALIDATION")
print("=" * 60)

if np.allclose(np.sum(loss_fn.probs, axis=1), 1.0):
    print("✓ Softmax probabilities sum to 1")
else:
    print("✗ Softmax incorrect")

if loss > 0:
    print("✓ Loss is positive")
else:
    print("✗ Loss incorrect")

print("=" * 60)