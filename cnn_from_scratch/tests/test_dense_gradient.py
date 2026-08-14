import numpy as np
from dense import Dense


def relative_error(a, b):
    numerator = np.max(np.abs(a - b))
    denominator = np.max(np.abs(a) + np.abs(b)) + 1e-12
    return numerator / denominator


print("=" * 60)
print("DENSE LAYER GRADIENT CHECK")
print("=" * 60)


# ------------------------------------------------------------
# 1. Small deterministic test data
# ------------------------------------------------------------

np.random.seed(42)

batch_size = 2
input_size = 4
output_size = 3

X = np.random.randn(batch_size, input_size)
dY = np.random.randn(batch_size, output_size)


# ------------------------------------------------------------
# 2. Create Dense layer
# ------------------------------------------------------------

dense = Dense(input_size, output_size)


# ------------------------------------------------------------
# 3. Forward + analytical backward
# ------------------------------------------------------------

Y = dense.forward(X)

dX = dense.backward(dY)

# Your Dense class stores these gradients
dW = dense.dW
db = dense.db


print("\nAnalytical gradients calculated.")


# ------------------------------------------------------------
# 4. Numerical gradient for W
# ------------------------------------------------------------

epsilon = 1e-5

numerical_dW = np.zeros_like(dense.W)

for i in range(dense.W.shape[0]):
    for j in range(dense.W.shape[1]):

        original_value = dense.W[i, j]

        # W + epsilon
        dense.W[i, j] = original_value + epsilon

        plus_output = dense.forward(X)
        loss_plus = np.sum(plus_output * dY)

        # W - epsilon
        dense.W[i, j] = original_value - epsilon

        minus_output = dense.forward(X)
        loss_minus = np.sum(minus_output * dY)

        # Central difference
        numerical_dW[i, j] = (
            loss_plus - loss_minus
        ) / (2 * epsilon)

        # Restore original value
        dense.W[i, j] = original_value


# ------------------------------------------------------------
# 5. Numerical gradient for b
# ------------------------------------------------------------

numerical_db = np.zeros_like(dense.b)

for j in range(dense.b.shape[1]):

    original_value = dense.b[0, j]

    # b + epsilon
    dense.b[0, j] = original_value + epsilon

    plus_output = dense.forward(X)
    loss_plus = np.sum(plus_output * dY)

    # b - epsilon
    dense.b[0, j] = original_value - epsilon

    minus_output = dense.forward(X)
    loss_minus = np.sum(minus_output * dY)

    # Central difference
    numerical_db[0, j] = (
        loss_plus - loss_minus
    ) / (2 * epsilon)

    # Restore original value
    dense.b[0, j] = original_value


# ------------------------------------------------------------
# 6. Numerical gradient for X
# ------------------------------------------------------------

numerical_dX = np.zeros_like(X)

for i in range(X.shape[0]):
    for j in range(X.shape[1]):

        original_value = X[i, j]

        # X + epsilon
        X[i, j] = original_value + epsilon

        plus_output = dense.forward(X)
        loss_plus = np.sum(plus_output * dY)

        # X - epsilon
        X[i, j] = original_value - epsilon

        minus_output = dense.forward(X)
        loss_minus = np.sum(minus_output * dY)

        # Central difference
        numerical_dX[i, j] = (
            loss_plus - loss_minus
        ) / (2 * epsilon)

        # Restore original value
        X[i, j] = original_value


# ------------------------------------------------------------
# 7. Compare gradients
# ------------------------------------------------------------

error_dW = relative_error(dW, numerical_dW)
error_db = relative_error(db, numerical_db)
error_dX = relative_error(dX, numerical_dX)


print("\n" + "=" * 60)
print("GRADIENT COMPARISON")
print("=" * 60)

print(f"\ndW relative error: {error_dW:.10e}")
print(f"db relative error: {error_db:.10e}")
print(f"dX relative error: {error_dX:.10e}")


# ------------------------------------------------------------
# 8. Validation
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("VALIDATION")
print("=" * 60)

threshold = 1e-5

if error_dW < threshold:
    print("✓ dW gradient correct")
else:
    print("✗ dW gradient incorrect")

if error_db < threshold:
    print("✓ db gradient correct")
else:
    print("✗ db gradient incorrect")

if error_dX < threshold:
    print("✓ dX gradient correct")
else:
    print("✗ dX gradient incorrect")

print("=" * 60)