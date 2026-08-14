import numpy as np
from cnn import CNN
from loss import SoftmaxCrossEntropy
from data_loader import load_fashion_mnist


def one_hot(y, num_classes=10):
    result = np.zeros((len(y), num_classes))
    result[np.arange(len(y)), y] = 1
    return result


def evaluate(model, X, y, batch_size=32):
    correct = 0

    for i in range(0, len(X), batch_size):
        X_batch = X[i:i + batch_size]
        y_batch = y[i:i + batch_size]

        logits = model.forward(X_batch)

        predictions = np.argmax(logits, axis=1)

        correct += np.sum(predictions == y_batch)

    return correct / len(X)


# ============================================================
# Load data
# ============================================================

X_train, y_train, X_test, y_test = load_fashion_mnist(
    "data/fashion-mnist_train.csv",
    "data/fashion-mnist_test.csv"
)

X_train = X_train[:5000]
y_train = y_train[:5000]

X_test = X_test[:1000]
y_test = y_test[:1000]


# ============================================================
# Model
# ============================================================

np.random.seed(42)

model = CNN()
loss_fn = SoftmaxCrossEntropy()

lr = 0.001
epochs = 3
batch_size = 32


# ============================================================
# History
# ============================================================

loss_history = []
train_accuracy_history = []
test_accuracy_history = []


# ============================================================
# Training
# ============================================================

for epoch in range(epochs):

    indices = np.random.permutation(len(X_train))

    X_train = X_train[indices]
    y_train = y_train[indices]

    total_loss = 0
    num_batches = len(X_train) // batch_size

    for batch in range(num_batches):

        start = batch * batch_size
        end = start + batch_size

        X_batch = X_train[start:end]
        y_batch = y_train[start:end]

        y_one_hot = one_hot(y_batch)

        # Forward
        logits = model.forward(X_batch)

        # Loss
        loss = loss_fn.forward(
            logits,
            y_one_hot
        )

        # Backward
        d_logits = loss_fn.backward()

        model.backward(d_logits)

        # Update
        model.conv1.W -= lr * model.conv1.dW
        model.conv1.b -= lr * model.conv1.db

        model.conv2.W -= lr * model.conv2.dW
        model.conv2.b -= lr * model.conv2.db

        model.fc1.W -= lr * model.fc1.dW
        model.fc1.b -= lr * model.fc1.db

        model.fc2.W -= lr * model.fc2.dW
        model.fc2.b -= lr * model.fc2.db

        total_loss += loss

    epoch_loss = total_loss / num_batches

    train_accuracy = evaluate(
        model,
        X_train,
        y_train,
        batch_size
    )

    test_accuracy = evaluate(
        model,
        X_test,
        y_test,
        batch_size
    )

    loss_history.append(epoch_loss)
    train_accuracy_history.append(train_accuracy)
    test_accuracy_history.append(test_accuracy)

    print("\n" + "=" * 50)
    print(f"Epoch {epoch + 1}/{epochs}")
    print(f"Loss:           {epoch_loss:.4f}")
    print(f"Train Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy:  {test_accuracy:.4f}")
    print("=" * 50)


# ============================================================
# Save metrics
# ============================================================

np.save("loss_history.npy", np.array(loss_history))
np.save(
    "train_accuracy_history.npy",
    np.array(train_accuracy_history)
)
np.save(
    "test_accuracy_history.npy",
    np.array(test_accuracy_history)
)

print("\nTraining complete.")