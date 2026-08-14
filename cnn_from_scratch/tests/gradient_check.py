import numpy as np

from conv import Conv2D


np.random.seed(42)

# ============================================================
# Small test input
# ============================================================

X = np.random.randn(1, 5, 5, 1)

conv = Conv2D(
    input_channels=1,
    num_filters=2,
    kernel_size=3,
    stride=1,
    padding=1
)

# Forward pass
output = conv.forward(X)

# Fake upstream gradient
d_output = np.random.randn(*output.shape)

# ============================================================
# Analytical gradients
# ============================================================

dX = conv.backward(d_output)
dW = conv.dW.copy()
db = conv.db.copy()

epsilon = 1e-5


# ============================================================
# Helper function
# ============================================================

def relative_error(analytical, numerical):

    numerator = np.max(
        np.abs(analytical - numerical)
    )

    denominator = np.max(
        np.abs(analytical) + np.abs(numerical)
    )

    return numerator / (denominator + 1e-8)


# ============================================================
# Numerical dW
# ============================================================

numerical_dW = np.zeros_like(conv.W)

for f in range(conv.num_filters):

    for i in range(conv.kernel_size):

        for j in range(conv.kernel_size):

            for c in range(conv.input_channels):

                original = conv.W[f, i, j, c]

                # W + epsilon
                conv.W[f, i, j, c] = original + epsilon

                plus_output = conv.forward(X)

                loss_plus = np.sum(
                    plus_output * d_output
                )

                # W - epsilon
                conv.W[f, i, j, c] = original - epsilon

                minus_output = conv.forward(X)

                loss_minus = np.sum(
                    minus_output * d_output
                )

                # Restore
                conv.W[f, i, j, c] = original

                numerical_dW[f, i, j, c] = (
                    loss_plus - loss_minus
                ) / (2 * epsilon)


# ============================================================
# Numerical db
# ============================================================

numerical_db = np.zeros_like(conv.b)

for f in range(conv.num_filters):

    original = conv.b[f]

    # b + epsilon
    conv.b[f] = original + epsilon

    plus_output = conv.forward(X)

    loss_plus = np.sum(
        plus_output * d_output
    )

    # b - epsilon
    conv.b[f] = original - epsilon

    minus_output = conv.forward(X)

    loss_minus = np.sum(
        minus_output * d_output
    )

    # Restore
    conv.b[f] = original

    numerical_db[f] = (
        loss_plus - loss_minus
    ) / (2 * epsilon)


# ============================================================
# Numerical dX
# ============================================================

numerical_dX = np.zeros_like(X)

for n in range(X.shape[0]):

    for i in range(X.shape[1]):

        for j in range(X.shape[2]):

            for c in range(X.shape[3]):

                original = X[n, i, j, c]

                # X + epsilon
                X[n, i, j, c] = original + epsilon

                plus_output = conv.forward(X)

                loss_plus = np.sum(
                    plus_output * d_output
                )

                # X - epsilon
                X[n, i, j, c] = original - epsilon

                minus_output = conv.forward(X)

                loss_minus = np.sum(
                    minus_output * d_output
                )

                # Restore
                X[n, i, j, c] = original

                numerical_dX[n, i, j, c] = (
                    loss_plus - loss_minus
                ) / (2 * epsilon)


# ============================================================
# Compare
# ============================================================

error_dW = relative_error(
    dW,
    numerical_dW
)

error_db = relative_error(
    db,
    numerical_db
)

error_dX = relative_error(
    dX,
    numerical_dX
)


# ============================================================
# Results
# ============================================================

print("=" * 60)
print("CONV2D COMPLETE GRADIENT CHECK")
print("=" * 60)

print("\ndW")
print("Maximum absolute difference:",
      np.max(np.abs(dW - numerical_dW)))
print("Relative error:", error_dW)

print("\ndb")
print("Maximum absolute difference:",
      np.max(np.abs(db - numerical_db)))
print("Relative error:", error_db)

print("\ndX")
print("Maximum absolute difference:",
      np.max(np.abs(dX - numerical_dX)))
print("Relative error:", error_dX)


# ============================================================
# Final result
# ============================================================

print("\n" + "=" * 60)
print("RESULT")
print("=" * 60)

threshold = 1e-5

dW_pass = error_dW < threshold
db_pass = error_db < threshold
dX_pass = error_dX < threshold

print("dW:", "PASS ✓" if dW_pass else "FAIL ✗")
print("db:", "PASS ✓" if db_pass else "FAIL ✗")
print("dX:", "PASS ✓" if dX_pass else "FAIL ✗")


if dW_pass and db_pass and dX_pass:

    print("\n✓ ALL GRADIENT CHECKS PASSED")

else:

    print("\n✗ GRADIENT CHECK FAILED")